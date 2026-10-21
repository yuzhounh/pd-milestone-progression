"""Step 15: multivariable odds ratios of the final predictors and forest plot."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from baseline_utils import (
    CURATED_FILENAME,
    baseline_correction_audit,
    load_curated_data,
    prepare_baseline_records,
)
from moca_utils import RAW_MOCA_FILENAME, moca_source_audit


warnings.filterwarnings("ignore")

SELECTED_FEATURES_PATH = "step_09_selected_features.csv"
FEATURE_MAPPING_PATH = "PPMI_feature_mapping.csv"
CURATED_PATH = Path(CURATED_FILENAME)


def load_features():
    selected = pd.read_csv(SELECTED_FEATURES_PATH)
    if selected["feature"].str.startswith("moca_").any():
        raise ValueError("MoCA source metadata cannot be selected as model features")
    return selected["feature"].tolist()


def load_feature_labels():
    mapping = pd.read_csv(FEATURE_MAPPING_PATH)
    return dict(zip(mapping["Feature"], mapping["Abbreviation"]))


FEATURES = load_features()
FEATURE_LABELS = load_feature_labels()


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_baseline_data():
    # Retain the source audits, then use the exact train-fitted/imputed feature
    # representation produced by Step 08 (including race indicators).
    curated = load_curated_data(CURATED_PATH)
    raw_moca = pd.read_csv(Path("tables") / RAW_MOCA_FILENAME, low_memory=False)
    baseline = prepare_baseline_records(
        curated, source_path=CURATED_PATH, raw_moca=raw_moca
    )
    baseline_correction_audit(baseline).to_csv(
        "step_15_baseline_corrections.csv", index=False
    )
    moca_source_audit(baseline).to_csv("step_15_moca_source_audit.csv", index=False)

    train = pd.read_csv("PPMI_4_train.csv")
    test = pd.read_csv("PPMI_4_test.csv")
    combined = pd.concat([train, test], ignore_index=True)
    missing = [feature for feature in FEATURES if feature not in combined.columns]
    if missing:
        raise ValueError(f"Missing features in Step 08 model data: {missing}")
    result = combined[["Target"] + FEATURES].copy()
    if result.isna().any().any():
        raise ValueError("Step 08 model data contain missing selected features")
    return result


def fit_multivariable_logistic(df_model):
    x = df_model[FEATURES]
    y = df_model["Target"].astype(int)

    binary_features = [
        feature
        for feature in x.columns
        if set(pd.to_numeric(x[feature], errors="raise").unique()).issubset({0, 1})
        and x[feature].nunique() == 2
    ]
    continuous_features = [
        feature for feature in x.columns if feature not in binary_features
    ]

    for feature in binary_features:
        observed_levels = sorted(x[feature].dropna().unique().tolist())
        if observed_levels != [0, 1]:
            raise ValueError(f"{feature} must be coded as 0/1, got {observed_levels}")

    scaler = StandardScaler()
    x_scaled = x.copy()
    x_scaled[continuous_features] = scaler.fit_transform(x[continuous_features])
    x_scaled = sm.add_constant(x_scaled)

    model = sm.Logit(y, x_scaled)
    result = model.fit(disp=False)

    summary = result.conf_int()
    summary["OR"] = result.params
    summary.columns = ["Lower CI", "Upper CI", "OR"]
    summary = np.exp(summary)
    summary["P-value"] = result.pvalues
    summary = summary.iloc[1:].copy()
    summary["Feature"] = summary.index
    summary["Display"] = (
        summary["Feature"].map(FEATURE_LABELS).fillna(summary["Feature"])
    )
    summary = summary.sort_values("OR", ascending=True)

    summary.to_csv("step_15_or_results.csv", index=False)
    return summary


def format_p_value(p_value):
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def plot_or_forest(summary):
    y_pos = np.arange(len(summary))
    or_vals = summary["OR"].to_numpy()
    low_ci = summary["Lower CI"].to_numpy()
    up_ci = summary["Upper CI"].to_numpy()
    labels = (
        summary["Display"]
        .where(
            summary["Feature"] != "SEX",
            "Sex (male vs female; 1 vs 0)",
        )
        .tolist()
    )

    color_risk = "#D81B60"
    color_protective = "#0072B2"
    point_colors = [color_protective if value < 1 else color_risk for value in or_vals]

    fig, (ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(4.6, 2.65),
        sharey=True,
        gridspec_kw={"width_ratios": [3.2, 1.25], "wspace": 0.04},
    )

    ax.errorbar(
        or_vals,
        y_pos,
        xerr=[or_vals - low_ci, up_ci - or_vals],
        fmt="none",
        ecolor="#7A7A7A",
        elinewidth=0.85,
        capsize=2.2,
        capthick=0.85,
        zorder=1,
    )
    ax.scatter(
        or_vals,
        y_pos,
        s=20,
        color=point_colors,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )

    ax.axvline(1, linestyle="--", color="#555555", linewidth=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_ylim(-0.6, len(summary) - 0.4)
    ax.set_xlim(0.30, 1.70)
    ax.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=0.7, length=2.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(ax.get_ylim())
    table_ax.axis("off")
    table_ax.text(
        0.03, len(summary) - 0.08, "OR (95% CI)", fontsize=6.2, ha="left", va="bottom"
    )
    table_ax.text(
        0.86, len(summary) - 0.08, "P", fontsize=6.2, ha="center", va="bottom"
    )

    for i, (_, row) in enumerate(summary.iterrows()):
        or_text = f"{row['OR']:.2f} ({row['Lower CI']:.2f}-{row['Upper CI']:.2f})"
        p_text = format_p_value(row["P-value"])
        table_ax.text(0.03, i, or_text, fontsize=5.9, ha="left", va="center")
        table_ax.text(0.86, i, p_text, fontsize=5.9, ha="center", va="center")

    fig.savefig("step_15_or_forest.pdf")
    plt.close(fig)


def main():
    set_publication_style()
    df_model = load_baseline_data()
    summary = fit_multivariable_logistic(df_model)
    plot_or_forest(summary)

    print("=== Step 15 OR results ===")
    print(
        summary[["Display", "Lower CI", "Upper CI", "OR", "P-value"]]
        .round(4)
        .to_string(index=False)
    )
    print(
        f"\nFull-cohort sample size after training-set-based imputation: {len(df_model)}"
    )
    print("Saved: step_15_or_forest.pdf")
    print("Saved: step_15_or_results.csv")


if __name__ == "__main__":
    main()
