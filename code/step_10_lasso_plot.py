"""Step 10: plot the LASSO cross-validation curve and coefficient paths from the step 08 outputs."""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs
from figure_style import MM_TO_INCH

fs.apply()
# Computer Modern math font, the same as the formulas in the manuscript (the lambda labels)
plt.rcParams["mathtext.fontset"] = "cm"


FEATURE_MAPPING_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "PPMI_feature_mapping.csv"
)


def load_feature_name_map(csv_path=FEATURE_MAPPING_CSV):
    df = pd.read_csv(csv_path)
    return dict(zip(df["Feature"], df["Abbreviation"]))


FEATURE_NAME_MAP = load_feature_name_map()

COLORS = {
    "positive": "#D81B60",
    "negative": "#1E88E5",
    "best": "#D55E00",
    "path": "#4D4D4D",
}


def map_feature_name(name):
    return FEATURE_NAME_MAP.get(name, name)


def save_pdf_png(fig, save_base):
    pdf_path = f"{save_base}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def plot_cv_and_coef_path(cv_scores_path, coef_path_csv, best_params_path, save_base):
    cv = pd.read_csv(cv_scores_path)
    coef_df = pd.read_csv(coef_path_csv)
    best = pd.read_csv(best_params_path).set_index("rule")

    best_c = best.loc["max", "best_C"]
    best_score = best.loc["max", "best_score"]
    best_lambda = 1 / best_c

    lambda_seq = cv["lambda"].values
    mean_scores = cv["mean_score"].values
    se_scores = (
        cv["se_score"].values
        if "se_score" in cv.columns
        else cv["std_score"].values / np.sqrt(5)
    )

    meta_cols = {"C", "lambda", "log_lambda"}
    feature_cols = [c for c in coef_df.columns if c not in meta_cols]
    coef_lambda_seq = coef_df["lambda"].values
    coefficients_path = coef_df[feature_cols].values

    feature_indices_to_plot = range(len(feature_cols))
    print(f"Figure 3B coefficient paths: all {len(feature_cols)} encoded variables")

    if True:
        # Equal-width panels, matching the reference LASSO figure without a path legend.
        fig, axes_grid = fs.axes_grid(1, 2)
        ax1, ax2 = axes_grid[0]

        ax1.plot(
            lambda_seq, mean_scores, "-", linewidth=1.2, marker="o", markersize=2.2
        )
        ax1.fill_between(
            lambda_seq,
            mean_scores - se_scores,
            mean_scores + se_scores,
            alpha=0.2,
        )
        best_lambda_line = ax1.axvline(
            best_lambda,
            color="r",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
        )
        ax1.plot(best_lambda, best_score, "ro", markersize=4.2)
        ax1.set_xscale("log")
        ax1.set_xlabel(r"$\lambda$")
        ax1.set_ylabel("AUPRC")
        fs.aligned_legend(
            ax1,
            [
                {
                    "name": r"Best $\lambda$ =",
                    "value": f"{best_lambda:.3f}",
                    "color": "red",
                    "kind": "line",
                    "linestyle": best_lambda_line.get_linestyle(),
                    "linewidth": best_lambda_line.get_linewidth(),
                },
            ],
            "lower left",
            handle_width=best_lambda_line.get_linewidth()
            * (
                3 * plt.rcParams["lines.dashed_pattern"][0]
                + 2 * plt.rcParams["lines.dashed_pattern"][1]
            ),
        )
        fs.panel_letter(ax1, "A")

        for i in feature_indices_to_plot:
            ax2.plot(
                coef_lambda_seq,
                coefficients_path[:, i],
                label=map_feature_name(feature_cols[i]).split(" (")[
                    0
                ],  # legend without units
                linewidth=1.1,
                alpha=0.85,
            )
        ax2.set_xscale("log")
        ax2.set_xlabel(r"$\lambda$")
        ax2.set_ylabel("Coefficient")
        ax2.axvline(best_lambda, color="r", linestyle="--", linewidth=0.9, alpha=0.4)
        fs.panel_letter(ax2, "B")

        for ax in (ax1, ax2):
            ax.set_xticks(
                [0.1, 1, 10, 100],
                [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"],
            )
            ax.set_xlim(right=10**2.5)
            ax.grid(alpha=0.25)

        save_pdf_png(fig, f"{save_base}_cv_coef_path")


def plot_lasso_coefficients(selected_features_csv, save_base):
    df = pd.read_csv(selected_features_csv)
    df = df[df["coefficient"] != 0].copy()
    df["feature_display"] = df["feature"].apply(map_feature_name)
    df = df.sort_values("coefficient", key=lambda x: x.abs(), ascending=True)

    coef_abs_max = df["coefficient"].abs().max()
    x_pad = coef_abs_max * 0.22
    x_min = df["coefficient"].min() - x_pad
    x_max = df["coefficient"].max() + x_pad

    fig, ax = plt.subplots(figsize=(120 * MM_TO_INCH, 62 * MM_TO_INCH))
    colors = [
        COLORS["positive"] if c > 0 else COLORS["negative"] for c in df["coefficient"]
    ]
    bars = ax.barh(
        np.arange(len(df)),
        df["coefficient"],
        color=colors,
        height=0.58,
        edgecolor="none",
        alpha=0.95,
    )

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["feature_display"], fontsize=6.4)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("LASSO coefficient")
    ax.axvline(0, color="0.25", linewidth=0.75)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.28)
    ax.set_axisbelow(True)

    offset = coef_abs_max * 0.035
    for bar, coef in zip(bars, df["coefficient"]):
        label_x = coef + offset if coef > 0 else coef - offset
        ha = "left" if coef > 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{coef:.4f}",
            va="center",
            ha=ha,
            fontsize=5.8,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["positive"], label="Risk-associated"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["negative"], label="Protective"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    ax.tick_params(axis="both", length=2.5, width=0.65, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    fig.subplots_adjust(left=0.31, right=0.98, top=0.96, bottom=0.18)
    save_pdf_png(fig, save_base)


def main():
    save_base = "step_10"
    print("Generating revised LASSO figures from existing step_08 outputs...")

    plot_cv_and_coef_path(
        cv_scores_path="step_08_cv_scores.csv",
        coef_path_csv="step_08_coef_path.csv",
        best_params_path="step_08_cv_best_params.csv",
        save_base=save_base,
    )

    plot_lasso_coefficients(
        selected_features_csv="step_08_selected_features.csv",
        save_base=f"{save_base}_lasso_coefficients",
    )


if __name__ == "__main__":
    main()
