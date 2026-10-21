"""Step 04: classification quality of the latent class models (AvePP, OCC, entropy) and follow-up summaries."""

import importlib.metadata
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from lca_utils import (
    DEFAULT_INIT_SEEDS,
    MEASUREMENT,
    build_discrete_time_matrix,
    build_posterior_table,
    calculate_entropy,
    class_expected_event_codes,
    fit_stable_model,
    load_lca_input,
)


INPUT_FILE = "PPMI_2_time_to_event.csv"
QUALITY_OUTPUT = "step_04_lca_classification_quality.csv"
MODEL_OUTPUT = "step_04_lca_model_details.csv"
EXPECTED_TIME_OUTPUT = "step_04_lca_expected_event_codes.csv"
FOLLOWUP_OUTPUT = "step_04_lca_followup_by_subtype.csv"
N_COMPONENTS = 2
MAX_ITER = 1000

warnings.filterwarnings("ignore")


def get_package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_quality_table(posterior: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    total_n = len(posterior)

    for subtype in ["Slow", "Rapid", "Overall"]:
        subset = (
            posterior
            if subtype == "Overall"
            else posterior[posterior["Subtype"] == subtype]
        )
        if subset.empty:
            continue

        max_posterior = subset["Subtype_Prob"]
        prevalence = len(subset) / total_n
        avepp = float(max_posterior.mean())
        occ = np.nan
        if subtype != "Overall" and 0.0 < prevalence < 1.0 and avepp < 1.0:
            occ = (avepp / (1.0 - avepp)) / (prevalence / (1.0 - prevalence))

        rows.append(
            {
                "Subtype": subtype,
                "N": int(len(subset)),
                "Percent": prevalence * 100,
                "Mean_assigned_class_posterior": avepp,
                "SD_assigned_class_posterior": max_posterior.std(ddof=1),
                "Median_max_posterior": max_posterior.median(),
                "Q1_max_posterior": max_posterior.quantile(0.25),
                "Q3_max_posterior": max_posterior.quantile(0.75),
                "Min_max_posterior": max_posterior.min(),
                "Odds_correct_classification": occ,
                "Max_posterior_lt_0_80_N": int((max_posterior < 0.80).sum()),
                "Max_posterior_lt_0_80_percent": (max_posterior < 0.80).mean() * 100,
                "Max_posterior_lt_0_90_N": int((max_posterior < 0.90).sum()),
                "Max_posterior_lt_0_90_percent": (max_posterior < 0.90).mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def build_followup_table(posterior: pd.DataFrame) -> pd.DataFrame:
    rapid = posterior.loc[posterior["Subtype"] == "Rapid", "max_followup"]
    slow = posterior.loc[posterior["Subtype"] == "Slow", "max_followup"]
    mann_whitney_p = float(mannwhitneyu(rapid, slow, alternative="two-sided").pvalue)

    year6_table = pd.crosstab(posterior["Subtype"], posterior["max_followup"] >= 6.0)
    year6_table = year6_table.reindex(
        index=["Rapid", "Slow"], columns=[False, True], fill_value=0
    )
    year6_p = float(chi2_contingency(year6_table, correction=False).pvalue)

    pooled_sd = np.sqrt(
        ((len(rapid) - 1) * rapid.var(ddof=1) + (len(slow) - 1) * slow.var(ddof=1))
        / (len(rapid) + len(slow) - 2)
    )
    standardized_mean_difference = float((rapid.mean() - slow.mean()) / pooled_sd)

    rows = []
    for subtype in ["Slow", "Rapid", "Overall"]:
        subset = (
            posterior
            if subtype == "Overall"
            else posterior[posterior["Subtype"] == subtype]
        )
        followup = subset["max_followup"]
        year6 = followup >= 6.0
        rows.append(
            {
                "Subtype": subtype,
                "N": int(len(subset)),
                "Followup_mean": followup.mean(),
                "Followup_SD": followup.std(ddof=1),
                "Followup_median": followup.median(),
                "Followup_Q1": followup.quantile(0.25),
                "Followup_Q3": followup.quantile(0.75),
                "Year6_followup_N": int(year6.sum()),
                "Year6_followup_percent": year6.mean() * 100,
                "Rapid_minus_slow_followup_SMD": standardized_mean_difference,
                "Mann_Whitney_p": mann_whitney_p,
                "Year6_chi_square_p": year6_p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Compute and save the classification-quality and follow-up tables."""
    frame, time_cols = load_lca_input(INPUT_FILE)
    matrix = build_discrete_time_matrix(frame, time_cols)
    print(
        f">>> N={len(matrix)}, event-history indicators={matrix.shape[1]}, "
        f"observed risk intervals={int(matrix.notna().sum().sum())}"
    )
    print(">>> Fitting censoring-aware 2-class discrete-time model...")
    stable_fit = fit_stable_model(matrix, N_COMPONENTS, max_iter=MAX_ITER)
    model = stable_fit.model
    probabilities = model.predict_proba(matrix)
    posterior = build_posterior_table(frame, matrix, model, time_cols)

    quality = build_quality_table(posterior)
    expected_times = class_expected_event_codes(model, time_cols)
    subtype_by_class = posterior.groupby("Subtype_Label")["Subtype"].first()
    expected_times["Subtype"] = expected_times["Class"].map(subtype_by_class)
    followup = build_followup_table(posterior)
    details = pd.DataFrame(
        [
            {
                "stepmix_version": get_package_version("stepmix"),
                "n_components": N_COMPONENTS,
                "measurement": MEASUREMENT,
                "event_history_encoding": "0=event-free;1=first event;NaN=post-event or right-censored",
                "n_independent_starts": len(DEFAULT_INIT_SEEDS),
                "max_iter": MAX_ITER,
                "best_seed": stable_fit.best_seed,
                "converged": bool(model.converged_),
                "iterations": int(model.n_iter_),
                "converged_starts": stable_fit.converged_starts,
                "mean_stability_ARI": stable_fit.mean_stability_ari,
                "min_stability_ARI": stable_fit.min_stability_ari,
                "N": len(matrix),
                "observed_risk_intervals": int(matrix.notna().sum().sum()),
                "n_parameters": int(model.n_parameters),
                "BIC": float(model.bic(matrix)),
                "AIC": float(model.aic(matrix)),
                "LogLikelihood": stable_fit.log_likelihood,
                "Entropy": calculate_entropy(probabilities),
            }
        ]
    )

    quality.to_csv(QUALITY_OUTPUT, index=False, encoding="utf-8-sig")
    details.to_csv(MODEL_OUTPUT, index=False, encoding="utf-8-sig")
    expected_times.to_csv(EXPECTED_TIME_OUTPUT, index=False, encoding="utf-8-sig")
    followup.to_csv(FOLLOWUP_OUTPUT, index=False, encoding="utf-8-sig")

    print("\nLCA model details")
    print("=" * 80)
    print(details.T.to_string(header=False))
    print("\nLCA classification quality")
    print("=" * 80)
    print(quality.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFollow-up by assigned subtype")
    print("=" * 80)
    print(followup.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\n>>> Saved:")
    for output in [QUALITY_OUTPUT, MODEL_OUTPUT, EXPECTED_TIME_OUTPUT, FOLLOWUP_OUTPUT]:
        print(f"    {output}")


if __name__ == "__main__":
    main()
