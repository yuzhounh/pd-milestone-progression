"""Step 11: combined ROC and precision-recall curves of the four classifiers on the test set."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import figstyle as fs

fs.apply()


MODEL_CONFIGS = {
    "SVM": {"file": "step_11_svm_predictions.csv", "color": "#1E88E5"},
    "LR": {"file": "step_11_lr_predictions.csv", "color": "#D55E00"},
    "RF": {"file": "step_11_rf_predictions.csv", "color": "#009E73"},
    "XGBoost": {"file": "step_11_xgboost_predictions.csv", "color": "#CC79A7"},
}


def load_predictions():
    probs = {}
    y_true_ref = None

    for name, config in MODEL_CONFIGS.items():
        df = pd.read_csv(config["file"])
        probs[name] = df["y_prob"].astype(float).values
        y_true = df["y_true"].astype(int).values
        if y_true_ref is None:
            y_true_ref = y_true
        elif not np.array_equal(y_true_ref, y_true):
            raise ValueError(f"y_true mismatch in {config['file']}")

    return y_true_ref, probs


def style_axis(ax):
    ax.grid(alpha=0.25)


def add_metric_legend(ax, metric, values, reference_label, reference_value, loc):
    rows = [
        {
            "name": n,
            "value": f"{values[n]:.3f}",
            "color": MODEL_CONFIGS[n]["color"],
            "kind": "line",
        }
        for n in MODEL_CONFIGS
    ]
    rows.append(
        {
            "name": reference_label,
            "value": "" if reference_value is None else f"{reference_value:.3f}",
            "color": "#4D4D4D",
            "kind": "line",
            "linestyle": "--",
            "linewidth": 0.9,
        }
    )
    fs.aligned_legend(ax, rows, loc, header=("Model", metric), size=5.4)


def main():
    y_test, probs = load_predictions()
    prevalence = y_test.mean()
    print(
        f"Loaded predictions: N={len(y_test)}, events={int(y_test.sum())}, prevalence={prevalence:.3f}"
    )

    fig, axes_grid = fs.axes_grid(1, 2)
    ax1, ax2 = axes_grid[0]

    auc_values = {}
    for name in MODEL_CONFIGS:
        auc_val = roc_auc_score(y_test, probs[name])
        auc_values[name] = auc_val
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        ax1.plot(
            fpr,
            tpr,
            color=MODEL_CONFIGS[name]["color"],
        )

    ax1.plot([0, 1], [0, 1], color="#4D4D4D", linewidth=0.9, linestyle="--")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("1 - Specificity")
    ax1.set_ylabel("Sensitivity")
    add_metric_legend(ax1, "AUROC", auc_values, "Chance", None, "lower right")
    fs.panel_letter(ax1, "A")
    style_axis(ax1)

    ap_values = {}
    for name in MODEL_CONFIGS:
        ap_val = average_precision_score(y_test, probs[name])
        ap_values[name] = ap_val
        precision, recall, _ = precision_recall_curve(y_test, probs[name])
        ax2.plot(
            recall,
            precision,
            color=MODEL_CONFIGS[name]["color"],
        )

    ax2.axhline(
        y=prevalence,
        color="#4D4D4D",
        linestyle="--",
        linewidth=0.9,
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    add_metric_legend(ax2, "AUPRC", ap_values, "Prevalence", prevalence, "upper right")
    fs.panel_letter(ax2, "B")
    style_axis(ax2)

    save_name = "step_11_AUC_curves"
    fig.savefig(f"{save_name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved: {save_name}.pdf")


if __name__ == "__main__":
    main()
