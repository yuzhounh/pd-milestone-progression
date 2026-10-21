"""Step 16: variance inflation factors and correlation heatmap of the final predictors."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from figure_style import (
    SUBPLOT_BOX_ASPECT,
    SUPPLEMENT_DPI,
    add_panel_label,
    apply_publication_style,
)

warnings.filterwarnings("ignore")

DPI = SUPPLEMENT_DPI

SELECTED_FEATURES_PATH = Path("step_09_selected_features.csv")
FEATURE_MAPPING_PATH = Path("PPMI_feature_mapping.csv")
TRAIN_PATH = Path("PPMI_4_train.csv")
TEST_PATH = Path("PPMI_4_test.csv")

SUBTYPE_PATH = Path("PPMI_3_patients_subtypes_lca_k2.csv")

VIF_CSV = "step_16_vif_results.csv"
CORR_CSV = "step_16_correlation_matrix.csv"
FIGURE_BASE = "step_16_vif_correlation"


def set_publication_style():
    apply_publication_style(DPI)


def load_selected_features(csv_path=SELECTED_FEATURES_PATH):
    selected = pd.read_csv(csv_path)
    return selected["feature"].dropna().tolist()


def load_feature_name_map(csv_path=FEATURE_MAPPING_PATH):
    mapping = pd.read_csv(csv_path)
    return dict(zip(mapping["Feature"], mapping["Abbreviation"]))


def load_complete_baseline_data():
    features = load_selected_features()
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing features in LASSO candidate data: {missing}")

    return data[features].dropna().copy()


def standardize(df):
    return (df - df.mean(axis=0)) / df.std(axis=0, ddof=0)


def calculate_single_vif(x_scaled, idx):
    y = x_scaled[:, idx]
    x_other = np.delete(x_scaled, idx, axis=1)
    x_design = np.column_stack([np.ones(x_other.shape[0]), x_other])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    y_hat = x_design @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return 1 / (1 - r_squared)


def calculate_vif(df):
    feature_name_map = load_feature_name_map()
    x_scaled = standardize(df)

    vif_df = pd.DataFrame(
        {
            "Feature": df.columns,
            "Display": [feature_name_map.get(col, col) for col in df.columns],
            "VIF": [
                calculate_single_vif(x_scaled.values, i)
                for i in range(x_scaled.shape[1])
            ],
        }
    ).sort_values("VIF", ascending=False)

    corr = x_scaled.corr()
    corr.index = [feature_name_map.get(col, col) for col in corr.index]
    corr.columns = [feature_name_map.get(col, col) for col in corr.columns]
    order = vif_df["Display"].tolist()
    corr = corr.loc[order, order]
    return vif_df, corr


def style_axis(ax):
    ax.tick_params(axis="both", length=2.4, width=0.65, pad=1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_vif_panel(ax, vif_df):
    plot_df = vif_df.sort_values("VIF", ascending=True)
    x_max = float(plot_df["VIF"].max()) * 1.18
    ax.barh(
        plot_df["Display"],
        plot_df["VIF"],
        color="#0072B2",
        edgecolor="black",
        linewidth=0.35,
        height=0.62,
    )
    for y, value in enumerate(plot_df["VIF"]):
        ax.text(value + 0.012, y, f"{value:.2f}", va="center", ha="left", fontsize=5.8)

    ax.set_xlabel("VIF")
    ax.set_ylabel("")
    ax.set_xlim(0, x_max)
    ax.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_axis(ax)


def draw_heatmap(ax, corr, cax=None):
    corr_values = corr.to_numpy()
    mask = np.triu(np.ones_like(corr_values, dtype=bool))
    masked = np.ma.array(corr_values, mask=mask)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")
    n = corr.shape[0]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(corr.index)
    ax.tick_params(axis="both", length=0, pad=1.5)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            if i > j:
                value = corr_values[i, j]
                text_color = "white" if abs(value) > 0.55 else "#222222"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.4,
                    color=text_color,
                )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if cax is not None:
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=6.0, length=2.2, width=0.65, pad=1.2)
        cbar.outline.set_linewidth(0.5)
    return im


def plot_combined(vif_df, corr):
    h_ax = 148.38 / 72.0
    w_vif = h_ax / SUBPLOT_BOX_ASPECT
    w_heat = h_ax  # Square correlation heatmap; the VIF panel retains 1.39:1.

    left_margin = 0.82
    gap = 1.05
    cbar_w = 0.08
    cbar_pad = 0.08
    right_margin = 0.45
    bottom_margin = 0.55
    top_margin = 0.35

    fig_w = left_margin + w_vif + gap + w_heat + cbar_pad + cbar_w + right_margin
    fig_h = bottom_margin + h_ax + top_margin

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    ax_vif = fig.add_axes(
        [left_margin / fig_w, bottom_margin / fig_h, w_vif / fig_w, h_ax / fig_h]
    )
    ax_heat = fig.add_axes(
        [
            (left_margin + w_vif + gap) / fig_w,
            bottom_margin / fig_h,
            w_heat / fig_w,
            h_ax / fig_h,
        ]
    )
    cax = ax_heat.inset_axes([1.04, 0.0, 0.045, 1.0], transform=ax_heat.transAxes)

    plot_vif_panel(ax_vif, vif_df)
    draw_heatmap(ax_heat, corr, cax=cax)
    add_panel_label(ax_vif, "A", x=-0.14, y=1.04, va="bottom")
    add_panel_label(ax_heat, "B", x=-0.14, y=1.04, va="bottom")

    fig.savefig(f"{FIGURE_BASE}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    set_publication_style()
    df = load_complete_baseline_data()
    vif_df, corr = calculate_vif(df)

    vif_df.to_csv(VIF_CSV, index=False, encoding="utf-8-sig")
    corr.to_csv(CORR_CSV, encoding="utf-8-sig")

    print(f"Complete cases: n={len(df)}")
    print("\nVIF diagnostic results:")
    print(vif_df[["Feature", "VIF"]].round(6).to_string(index=False))

    plot_combined(vif_df, corr)

    print(f"\nSaved: {FIGURE_BASE}.pdf")
    print(f"Saved: {VIF_CSV}")
    print(f"Saved: {CORR_CSV}")


if __name__ == "__main__":
    main()
