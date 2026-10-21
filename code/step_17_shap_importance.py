"""Step 17: compare SHAP feature importance across the four classifiers (bar and beeswarm panels)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from shap.plots.colors import red_blue

from figure_style import (
    MM_TO_INCH,
    SUBPLOT_BOX_ASPECT,
    SUPPLEMENT_DPI,
    add_panel_label,
    apply_publication_style,
)

DPI = SUPPLEMENT_DPI
OUTPUT_BASE = "step_17_shap_importance"
BEESWARM_OUTPUT = "step_17_shap_beeswarm.pdf"
FEATURE_MAPPING_PATH = Path("PPMI_feature_mapping.csv")
SELECTED_FEATURES_PATH = Path("step_09_selected_features.csv")
TRAIN_PATH = Path("PPMI_4_train.csv")
# Collect the training-set SHAP results of the step 11 models; all four use a random
# subsample of at most 500 training participants (seed 42).
MODEL_FILES = {
    "LR": Path("step_11_lr_feature_importance.csv"),
    "RF": Path("step_11_rf_feature_importance.csv"),
    "SVM": Path("step_11_svm_feature_importance.csv"),
    "XGBoost": Path("step_11_xgboost_feature_importance.csv"),
}
SHAP_VALUE_FILES = {
    "LR": Path("step_11_lr_shap_values.csv"),
    "RF": Path("step_11_rf_shap_values.csv"),
    "SVM": Path("step_11_svm_shap_values.csv"),
    "XGBoost": Path("step_11_xgboost_shap_values.csv"),
}

FEATURE_ALIASES = {
    "RBD": "RBDSQ",
    "rem": "RBDSQ",
    "mean striatal SBR": "Mean striatal SBR",
    "MIA Striatum Mean": "Mean striatal SBR",
    "MIA_STRIATUM_mean": "Mean striatal SBR",
    "SEX": "Sex",
    "scopa": "SCOPA-AUT",
    "DVT_SFTANIM": "SVF",
    "upsit_pctl": "UPSIT (percentile)",
    "UPSIT percentile": "UPSIT (percentile)",
    "UPSIT": "UPSIT (percentile)",
    "EDUCYRS": "Education",
}

BAR_COLOR = "#0072B2"


def set_publication_style():
    apply_publication_style(DPI)
    plt.rcParams.update({"savefig.bbox": "tight", "savefig.pad_inches": 0.04})


def load_feature_name_map(csv_path=FEATURE_MAPPING_PATH):
    mapping = pd.read_csv(csv_path)
    feature_map = dict(zip(mapping["Feature"], mapping["Abbreviation"]))
    abbreviation_map = dict(zip(mapping["Abbreviation"], mapping["Abbreviation"]))
    feature_map.update(abbreviation_map)
    feature_map.update(FEATURE_ALIASES)
    return feature_map


def load_feature_order(csv_path=SELECTED_FEATURES_PATH):
    selected = pd.read_csv(csv_path)
    if "feature" not in selected.columns:
        raise ValueError(f"{csv_path} does not contain a feature column.")

    feature_name_map = load_feature_name_map()
    feature_order = (
        selected["feature"].map(feature_name_map).fillna(selected["feature"]).tolist()
    )
    if len(feature_order) != len(set(feature_order)):
        raise ValueError("Selected features are not unique after display-name mapping.")
    return feature_order


def load_shap_importance(model_name, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file for {model_name}: {path}")

    feature_name_map = load_feature_name_map()

    if model_name == "SVM" and {"Display", "Mean_abs_SHAP"}.issubset(
        pd.read_csv(path, nrows=0).columns
    ):
        df = pd.read_csv(path)
        out = df[["Display", "Mean_abs_SHAP"]].rename(
            columns={"Display": "Feature", "Mean_abs_SHAP": "SHAP"}
        )
    else:
        df = pd.read_csv(path, index_col=0)
        if "SHAP" not in df.columns:
            raise ValueError(f"{path} does not contain a SHAP column.")
        out = df[["SHAP"]].rename_axis("Feature").reset_index()

    out = out.assign(
        Model=model_name,
        Feature=lambda x: x["Feature"].map(feature_name_map).fillna(x["Feature"]),
    )
    return out[["Model", "Feature", "SHAP"]]


def build_combined_table():
    feature_order = load_feature_order()
    frames = [load_shap_importance(model, path) for model, path in MODEL_FILES.items()]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["Feature"].isin(feature_order)].copy()

    expected = set(feature_order)
    for model in MODEL_FILES:
        observed = set(combined.loc[combined["Model"].eq(model), "Feature"])
        missing = expected - observed
        if missing:
            raise ValueError(f"{model} is missing SHAP values for: {sorted(missing)}")

    combined["Feature"] = pd.Categorical(
        combined["Feature"], categories=feature_order, ordered=True
    )
    combined = combined.sort_values(["Model", "Feature"]).reset_index(drop=True)
    combined.to_csv(f"{OUTPUT_BASE}.csv", index=False)
    return combined


def style_axis(ax):
    ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=0.7, length=2.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def plot_panel(ax, data, model_name, panel_label):
    model_df = data[data["Model"].eq(model_name)].copy()
    model_df = model_df.sort_values("SHAP", ascending=True)

    ax.barh(
        model_df["Feature"].astype(str),
        model_df["SHAP"],
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.35,
        height=0.66,
    )
    ax.set_title(model_name, pad=5)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")
    add_panel_label(ax, panel_label, x=-0.13, y=1.04, va="bottom")
    style_axis(ax)


def plot_combined(data):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(170 * MM_TO_INCH, 128 * MM_TO_INCH),
        dpi=DPI,
        constrained_layout=True,
    )
    panels = ["A", "B", "C", "D"]

    for ax, panel_label, model_name in zip(axes.ravel(), panels, MODEL_FILES.keys()):
        ax.set_box_aspect(SUBPLOT_BOX_ASPECT)
        plot_panel(ax, data, model_name, panel_label)

    fig.savefig(f"{OUTPUT_BASE}.pdf")
    plt.close(fig)


def load_standardized_training_features():
    selected = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].dropna().tolist()
    mapping = load_feature_name_map()
    display_names = [mapping.get(feature, feature) for feature in selected]
    training = pd.read_csv(TRAIN_PATH)
    values = StandardScaler().fit_transform(training[selected])
    return values, display_names


def load_beeswarm_data(model_name, path, standardized_values, display_names):
    shap_values = pd.read_csv(path)
    # Accept saved SHAP results carrying the previous presentation labels.
    shap_values = shap_values.rename(columns=load_feature_name_map())
    if shap_values.columns.tolist() != display_names:
        raise ValueError(
            f"{path.name} columns do not match the selected-feature display order."
        )

    np.random.seed(42)
    sample_indices = np.random.choice(
        len(standardized_values), len(shap_values), replace=False
    )
    sampled_values = standardized_values[sample_indices]
    return shap_values.to_numpy(dtype=float), sampled_values


def plot_beeswarm_panel(
    ax, shap_values, feature_values, feature_names, model_name, panel_label
):
    order = np.argsort(np.mean(np.abs(shap_values), axis=0))
    rng = np.random.default_rng(42)

    for row, feature_index in enumerate(order):
        values = feature_values[:, feature_index]
        color_values = pd.Series(values).rank(method="average", pct=True).to_numpy()
        jitter = rng.normal(0, 0.085, size=len(values))
        ax.scatter(
            shap_values[:, feature_index],
            row + jitter,
            c=color_values,
            cmap=red_blue,
            norm=Normalize(0, 1),
            s=9 if len(values) <= 10 else 5,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="#777777", linewidth=0.65)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([feature_names[index] for index in order])
    ax.set_xlabel("SHAP value")
    ax.set_title(model_name, pad=4)
    ax.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.28)
    ax.set_axisbelow(True)
    ax.set_box_aspect(SUBPLOT_BOX_ASPECT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, panel_label, x=-0.13, y=1.04, va="bottom")


def plot_combined_beeswarm():
    standardized_values, display_names = load_standardized_training_features()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(170 * MM_TO_INCH, 128 * MM_TO_INCH),
        dpi=DPI,
        constrained_layout=True,
    )

    for ax, panel_label, (model_name, path) in zip(
        axes.ravel(), ["A", "B", "C", "D"], SHAP_VALUE_FILES.items()
    ):
        shap_values, feature_values = load_beeswarm_data(
            model_name, path, standardized_values, display_names
        )
        plot_beeswarm_panel(
            ax,
            shap_values,
            feature_values,
            display_names,
            model_name,
            panel_label,
        )

    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=red_blue),
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.62,
        fraction=0.025,
        aspect=80,
        pad=0.02,
    )
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Feature value", labelpad=0)
    colorbar.ax.tick_params(length=0, pad=1.2)
    colorbar.outline.set_visible(False)

    fig.savefig(BEESWARM_OUTPUT)
    plt.close(fig)


def main():
    set_publication_style()
    combined = build_combined_table()
    plot_combined(combined)
    plot_combined_beeswarm()
    print(f"Saved: {OUTPUT_BASE}.pdf")
    print(f"Saved: {OUTPUT_BASE}.csv")
    print(f"Saved: {BEESWARM_OUTPUT}")


if __name__ == "__main__":
    main()
