"""Step 03: select the number of latent classes (k = 1-6) by AIC, BIC and entropy."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lca_utils import (
    DEFAULT_INIT_SEEDS,
    MEASUREMENT,
    build_discrete_time_matrix,
    calculate_entropy,
    fit_stable_model,
    load_lca_input,
)


INPUT_FILE = "PPMI_2_time_to_event.csv"
RESULT_OUTPUT = "step_03_lca_model_selection.csv"
PLOT_OUTPUT = "step_03_lca_model_selection_plot.pdf"
COMPONENTS = range(1, 7)

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = True


def main() -> None:
    """Fit the candidate latent class models and save the comparison table and plot."""
    print(">>> Fitting censoring-aware discrete-time latent class models (k=1..6)...")
    frame, time_cols = load_lca_input(INPUT_FILE)
    matrix = build_discrete_time_matrix(frame, time_cols)
    observed_cells = int(matrix.notna().sum().sum())
    total_cells = int(matrix.size)
    print(
        f">>> N={len(matrix)}, event-history indicators={matrix.shape[1]}, "
        f"observed risk intervals={observed_cells}/{total_cells}"
    )

    results: list[dict[str, object]] = []
    for n_components in COMPONENTS:
        print(
            f">>> Fitting k={n_components} with {len(DEFAULT_INIT_SEEDS)} independent starts..."
        )
        stable_fit = fit_stable_model(matrix, n_components)
        model = stable_fit.model
        probabilities = model.predict_proba(matrix)
        labels = model.predict(matrix).astype(int)
        counts = np.bincount(labels, minlength=n_components)
        class_avepp = [
            probabilities[labels == idx, idx].mean() for idx in range(n_components)
        ]

        results.append(
            {
                "K": n_components,
                "Measurement": MEASUREMENT,
                "N": len(matrix),
                "Observed_risk_intervals": observed_cells,
                "N_parameters": int(model.n_parameters),
                "LogLikelihood": stable_fit.log_likelihood,
                "BIC": float(model.bic(matrix)),
                "AIC": float(model.aic(matrix)),
                "Entropy": calculate_entropy(probabilities),
                "Class_sizes": ";".join(str(int(value)) for value in counts),
                "Class_proportions": ";".join(
                    f"{value / len(matrix):.6f}" for value in counts
                ),
                "Class_AvePP": ";".join(f"{value:.6f}" for value in class_avepp),
                "Converged": bool(model.converged_),
                "Iterations": int(model.n_iter_),
                "Best_seed": stable_fit.best_seed,
                "Converged_starts": stable_fit.converged_starts,
                "Total_starts": len(DEFAULT_INIT_SEEDS),
                "Mean_stability_ARI": stable_fit.mean_stability_ari,
                "Min_stability_ARI": stable_fit.min_stability_ari,
            }
        )

    result = pd.DataFrame(results)
    result.to_csv(RESULT_OUTPUT, index=False, encoding="utf-8-sig")

    display_columns = [
        "K",
        "BIC",
        "AIC",
        "Entropy",
        "Class_sizes",
        "Class_AvePP",
        "Converged_starts",
        "Mean_stability_ARI",
        "Min_stability_ARI",
    ]
    print(
        "\n"
        + result[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    best_k_bic = int(result.loc[result["BIC"].idxmin(), "K"])
    best_k_aic = int(result.loc[result["AIC"].idxmin(), "K"])
    print(f"\n>>> BIC-preferred solution: k={best_k_bic}")
    print(f">>> AIC-preferred solution: k={best_k_aic}")

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(result["K"], result["BIC"], marker="o", label="BIC", linewidth=2)
    axis.plot(result["K"], result["AIC"], marker="s", label="AIC", linewidth=2)
    axis.set_xlabel("Number of classes (k)", fontsize=12)
    axis.set_ylabel("Information criterion", fontsize=12)
    axis.set_title("Censoring-aware latent class model selection", fontsize=13)
    axis.set_xticks(list(COMPONENTS))
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_OUTPUT, bbox_inches="tight")
    plt.close(fig)

    print(f">>> Saved: {RESULT_OUTPUT}")
    print(f">>> Saved: {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
