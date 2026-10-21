"""Check nonlinear functional forms and influential observations for final LR predictors."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from figure_style import SUBPLOT_BOX_ASPECT


CODE_DIR = Path(__file__).resolve().parent
SELECTED_FEATURES_PATH = CODE_DIR / "step_09_selected_features.csv"
TRAIN_PATH = CODE_DIR / "PPMI_4_train.csv"

DISPLAY_NAMES = {
    "DVT_SDM": "SDMT",
    "EDUCYRS": "Education",
    "upsit_pctl": "UPSIT (percentile)",
    "orthostasis": "OH",
    "age_at_visit": "Age",
    "MSEADLG": "S&E-ADL",
    "updrs1_score": "MDS-UPDRS Part I",
    "updrs2_score": "MDS-UPDRS Part II",
    "updrs3_score": "MDS-UPDRS Part III",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = True


def fit_model(formula: str, data: pd.DataFrame):
    return smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
    ).fit()


def main(*, save_results: bool = True) -> None:
    selected = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()
    data = pd.read_csv(TRAIN_PATH)[["PATNO", "Target", *selected]].dropna().copy()
    if not selected:
        raise ValueError("No final predictors were selected")
    if data["PATNO"].duplicated().any():
        raise ValueError("Functional-form input contains duplicate PATNO values")

    linear_formula = "Target ~ " + " + ".join(selected)
    linear_model = fit_model(linear_formula, data)
    test_rows = []
    spline_models = {}
    binary_features = {
        feature
        for feature in selected
        if data[feature].nunique() == 2
        and set(pd.to_numeric(data[feature], errors="raise").unique()).issubset({0, 1})
    }

    for feature in selected:
        if feature in binary_features:
            spline_models[feature] = None
            test_rows.append(
                {
                    "feature": feature,
                    "display": DISPLAY_NAMES.get(feature, feature),
                    "n": len(data),
                    "linear_aic": linear_model.aic,
                    "spline_aic": np.nan,
                    "likelihood_ratio_chi2": np.nan,
                    "df": 0,
                    "p_nonlinearity": np.nan,
                    "retain_linear_form": True,
                }
            )
            continue
        other_terms = [value for value in selected if value != feature]
        spline_formula = f"Target ~ cr({feature}, df=4)" + (
            " + " + " + ".join(other_terms) if other_terms else ""
        )
        spline_model = fit_model(spline_formula, data)
        spline_models[feature] = spline_model
        lr_statistic = 2 * (spline_model.llf - linear_model.llf)
        degrees_freedom = int(round(spline_model.df_model - linear_model.df_model))
        p_value = float(chi2.sf(max(lr_statistic, 0), degrees_freedom))
        test_rows.append(
            {
                "feature": feature,
                "display": DISPLAY_NAMES.get(feature, feature),
                "n": len(data),
                "linear_aic": linear_model.aic,
                "spline_aic": spline_model.aic,
                "likelihood_ratio_chi2": lr_statistic,
                "df": degrees_freedom,
                "p_nonlinearity": p_value,
                "retain_linear_form": p_value >= 0.05,
            }
        )

    tests = pd.DataFrame(test_rows)
    if save_results:
        tests.to_csv(CODE_DIR / "step_09_nonlinearity_tests.csv", index=False)

    influence = linear_model.get_influence(observed=True).summary_frame()
    diagnostic = pd.DataFrame(
        {
            "PATNO": data["PATNO"].to_numpy(),
            "cooks_d": influence["cooks_d"].to_numpy(),
            "leverage": influence["hat_diag"].to_numpy(),
            "standardized_residual": influence["standard_resid"].to_numpy(),
        }
    )
    n = len(data)
    parameter_count = int(linear_model.df_model) + 1
    cook_threshold = 4 / n
    leverage_threshold = 2 * parameter_count / n
    diagnostic["high_cooks_d"] = diagnostic["cooks_d"] > cook_threshold
    diagnostic["high_leverage"] = diagnostic["leverage"] > leverage_threshold
    diagnostic["large_residual"] = diagnostic["standardized_residual"].abs() > 3
    # High leverage is reported separately; exclusion sensitivity is limited to
    # observations with outcome influence (Cook's D) or an extreme residual.
    diagnostic["influential"] = diagnostic[["high_cooks_d", "large_residual"]].any(
        axis=1
    )
    if save_results:
        diagnostic.sort_values("cooks_d", ascending=False).to_csv(
            CODE_DIR / "step_09_influence_diagnostics.csv", index=False
        )

    if save_results:
        pd.DataFrame(
            [
                {
                    "n": n,
                    "parameter_count": parameter_count,
                    "cooks_d_threshold": cook_threshold,
                    "high_cooks_d_n": int(diagnostic["high_cooks_d"].sum()),
                    "leverage_threshold": leverage_threshold,
                    "high_leverage_n": int(diagnostic["high_leverage"].sum()),
                    "standardized_residual_threshold": 3,
                    "large_residual_n": int(diagnostic["large_residual"].sum()),
                    "influence_exclusion_n": int(diagnostic["influential"].sum()),
                }
            ]
        ).to_csv(CODE_DIR / "step_09_influence_summary.csv", index=False)

    sensitivity_data = data.loc[~diagnostic["influential"].to_numpy()].copy()
    sensitivity_model = fit_model(linear_formula, sensitivity_data)
    sensitivity_rows = []
    for feature in selected:
        main_coefficient = float(linear_model.params[feature])
        sensitivity_coefficient = float(sensitivity_model.params[feature])
        sensitivity_rows.append(
            {
                "feature": feature,
                "display": DISPLAY_NAMES.get(feature, feature),
                "main_n": len(data),
                "sensitivity_n": len(sensitivity_data),
                "excluded_n": len(data) - len(sensitivity_data),
                "main_coefficient": main_coefficient,
                "sensitivity_coefficient": sensitivity_coefficient,
                "main_odds_ratio": np.exp(main_coefficient),
                "sensitivity_odds_ratio": np.exp(sensitivity_coefficient),
                "coefficient_sign_consistent": (
                    np.sign(main_coefficient) == np.sign(sensitivity_coefficient)
                ),
            }
        )
    if save_results:
        pd.DataFrame(sensitivity_rows).to_csv(
            CODE_DIR / "step_09_influence_sensitivity.csv", index=False
        )

    n_columns = min(4, len(selected))
    n_rows = math.ceil(len(selected) / n_columns)
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(2 * n_columns, 2.1 * n_rows), squeeze=False
    )
    reference = {feature: float(data[feature].median()) for feature in selected}
    for panel_index, (axis, feature) in enumerate(zip(axes.ravel(), selected)):
        if feature in binary_features:
            grid = np.array([0.0, 1.0])
        else:
            lower, upper = data[feature].quantile([0.01, 0.99])
            grid = np.linspace(lower, upper, 150)
        prediction_data = pd.DataFrame(
            {name: np.repeat(value, len(grid)) for name, value in reference.items()}
        )
        prediction_data[feature] = grid
        linear_prediction = linear_model.get_prediction(prediction_data).summary_frame()

        axis.plot(
            grid,
            linear_prediction["mean"],
            linestyle="-" if feature in binary_features else "--",
            marker="o" if feature in binary_features else None,
            linewidth=1.6,
            color="#666666",
            label="Linear",
        )
        if feature not in binary_features:
            spline_prediction = (
                spline_models[feature].get_prediction(prediction_data).summary_frame()
            )
            axis.plot(
                grid,
                spline_prediction["mean"],
                linewidth=2.0,
                color="#2468A2",
                label="Restricted cubic spline",
            )
            axis.fill_between(
                grid,
                spline_prediction["mean_ci_lower"].to_numpy(),
                spline_prediction["mean_ci_upper"].to_numpy(),
                color="#2468A2",
                alpha=0.16,
                linewidth=0,
            )
        axis.set_xlabel(DISPLAY_NAMES.get(feature, feature), fontsize=9)
        if panel_index % n_columns == 0:
            axis.set_ylabel("Adjusted probability", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.set_box_aspect(SUBPLOT_BOX_ASPECT)
        if feature in binary_features:
            axis.set_xticks([0, 1])
            axis.set_title("Binary indicator", fontsize=9)
        else:
            p_value = float(
                tests.loc[tests["feature"] == feature, "p_nonlinearity"].iloc[0]
            )
            axis.set_title(f"Nonlinearity $P$ = {p_value:.3f}", fontsize=9)
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in axes.ravel()[len(selected) :]:
        axis.set_visible(False)
    legend_items = {}
    for axis in axes.ravel()[: len(selected)]:
        handles, labels = axis.get_legend_handles_labels()
        legend_items.update(zip(labels, handles))
    fig.legend(
        list(legend_items.values()),
        list(legend_items),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(
        CODE_DIR / "step_09_functional_form_diagnostics.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Nonlinearity tests")
    print(
        tests[["feature", "likelihood_ratio_chi2", "df", "p_nonlinearity"]].to_string(
            index=False
        )
    )
    print(
        f"Influence thresholds: Cook's D > {cook_threshold:.6f}; "
        f"leverage > {leverage_threshold:.6f}; |standardized residual| > 3"
    )
    print(
        f"Flagged {int(diagnostic['influential'].sum())}/{n} observations; "
        + (
            "coefficient-sign sensitivity saved."
            if save_results
            else "saved numerical results preserved."
        )
    )


if __name__ == "__main__":
    main()
