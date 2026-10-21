"""Calibration and clinical-utility plots for held-out test-set predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

import figstyle as fs
from prediction_utils import load_aligned_prediction_frames

CODE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = CODE_DIR / "step_20_decision_curve_results.csv"
OUTPUT_IMPACT_CSV = CODE_DIR / "step_20_clinical_impact_results.csv"
OUTPUT_PDF = CODE_DIR / "step_20_model_calibration_utility.pdf"

MODEL_FILES = {
    "LR": CODE_DIR / "step_11_lr_predictions.csv",
    "SVM": CODE_DIR / "step_11_svm_predictions.csv",
    "RF": CODE_DIR / "step_11_rf_predictions.csv",
    "XGBoost": CODE_DIR / "step_11_xgboost_predictions.csv",
}

MODEL_COLORS = {
    "LR": "#1E88E5",
    "SVM": "#1E88E5",
    "RF": "#009E73",
    "XGBoost": "#CC79A7",
}
CLASS_COLORS = {"Slow": "#1E88E5", "Rapid": "#D81B60"}

fs.apply()


def load_predictions():
    frames = load_aligned_prediction_frames(MODEL_FILES)
    predictions = {}
    y_reference = next(iter(frames.values()))["y_true"].to_numpy(dtype=int)

    for model, frame in frames.items():
        calibration_methods = frame["calibration_method"].dropna().unique()
        if not (
            len(calibration_methods) == 1
            and calibration_methods[0] == "platt_10fold_oof"
        ):
            raise ValueError(
                f"{MODEL_FILES[model].name} must use 10-fold OOF Platt calibration"
            )

        y_prob = frame["y_prob"].astype(float).to_numpy()
        if not np.isfinite(y_prob).all() or np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError(
                f"{MODEL_FILES[model].name} contains invalid probabilities"
            )

        predictions[model] = y_prob

    return y_reference, predictions


def calculate_net_benefit(y_true, y_prob, thresholds):
    n = len(y_true)
    values = []
    for threshold in thresholds:
        predicted_positive = y_prob >= threshold
        true_positive = np.sum((y_true == 1) & predicted_positive)
        false_positive = np.sum((y_true == 0) & predicted_positive)
        values.append(
            (true_positive / n) - (false_positive / n) * threshold / (1 - threshold)
        )
    return np.asarray(values)


def build_results(y_true, predictions, thresholds):
    prevalence = float(np.mean(y_true))
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    rows = []

    for model, y_prob in predictions.items():
        model_net_benefit = calculate_net_benefit(y_true, y_prob, thresholds)
        for threshold, model_nb, treat_all_nb in zip(
            thresholds, model_net_benefit, treat_all
        ):
            rows.append(
                {
                    "threshold_probability": threshold,
                    "model": model,
                    "model_net_benefit": model_nb,
                    "treat_all_net_benefit": treat_all_nb,
                    "treat_none_net_benefit": 0.0,
                    "better_than_treat_all": model_nb > treat_all_nb,
                    "better_than_treat_none": model_nb > 0,
                }
            )

    return pd.DataFrame(rows)


def build_clinical_impact(y_true, y_prob, thresholds):
    rows = []
    n = len(y_true)
    for threshold in thresholds:
        predicted_positive = y_prob >= threshold
        true_positive = np.sum((y_true == 1) & predicted_positive)
        rows.append(
            {
                "threshold_probability": threshold,
                "high_risk_per_1000": 1000 * np.sum(predicted_positive) / n,
                "high_risk_with_outcome_per_1000": 1000 * true_positive / n,
            }
        )
    return pd.DataFrame(rows)


def plot_results(results, clinical_impact, y_true, predictions):
    figure, axes = fs.axes_grid(2, 2)
    calibration_axis, distribution_axis, decision_axis, impact_axis = axes.ravel()

    # A: calibration curve for the final LR model.
    lr_probabilities = predictions["LR"]
    calibration_limit = min(1.0, max(0.6, float(np.max(lr_probabilities)) + 0.05))
    calibration_axis.plot(
        [0, calibration_limit],
        [0, calibration_limit],
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        label="Ideal",
    )
    observed, predicted = calibration_curve(
        y_true, lr_probabilities, n_bins=8, strategy="quantile"
    )
    calibration_axis.plot(
        predicted,
        observed,
        marker="o",
        markersize=3.0,
        linewidth=1.1,
        color=MODEL_COLORS["LR"],
        label="LR",
    )
    calibration_axis.set_xlim(0, calibration_limit)
    calibration_axis.set_ylim(0, calibration_limit)
    calibration_axis.set_xlabel("Mean predicted probability")
    calibration_axis.set_ylabel("Assigned rapid-class proportion")
    calibration_axis.set_title("Calibration")
    calibration_axis.legend(frameon=True, ncol=1, loc="upper left")
    calibration_axis.grid(alpha=0.25)
    fs.panel_letter(calibration_axis, "A")

    # B: LR probability distributions normalized within each assigned class.
    histogram_limit = min(1.0, max(0.6, float(np.max(lr_probabilities)) + 0.05))
    bins = np.linspace(0, histogram_limit, 17)
    distribution_axis.hist(
        lr_probabilities[y_true == 0],
        bins=bins,
        density=True,
        alpha=0.55,
        color=CLASS_COLORS["Slow"],
        edgecolor="#333333",
        linewidth=0.35,
        label="Slow",
    )
    distribution_axis.hist(
        lr_probabilities[y_true == 1],
        bins=bins,
        density=True,
        alpha=0.55,
        color=CLASS_COLORS["Rapid"],
        edgecolor="#333333",
        linewidth=0.35,
        label="Rapid",
    )
    distribution_axis.set_xlim(0, histogram_limit)
    distribution_axis.set_xlabel("Predicted probability")
    distribution_axis.set_ylabel("Density")
    distribution_axis.set_title("Predicted-probability distribution")
    distribution_axis.legend(frameon=True, ncol=1, loc="upper right")
    distribution_axis.grid(alpha=0.25)
    fs.panel_letter(distribution_axis, "B")

    # C: decision curve analysis for the final LR model.
    reference = results.loc[results["model"] == "LR"]
    decision_axis.plot(
        reference["threshold_probability"],
        reference["model_net_benefit"],
        color=MODEL_COLORS["LR"],
        linewidth=1.15,
        label="LR",
    )
    decision_axis.plot(
        reference["threshold_probability"],
        reference["treat_all_net_benefit"],
        color="#555555",
        linestyle=(0, (1, 2)),
        linewidth=1.1,
        label="Treat all",
    )
    decision_axis.axhline(
        0,
        color="#999999",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        label="Treat none",
    )
    decision_axis.set_xlim(0.01, 0.50)
    decision_axis.set_ylim(-0.08, 0.20)
    decision_axis.set_xlabel("Threshold probability")
    decision_axis.set_ylabel("Net benefit")
    decision_axis.set_title("Decision curve analysis")
    decision_axis.grid(alpha=0.25)
    decision_axis.legend(frameon=True, ncol=1, loc="upper right")
    fs.panel_letter(decision_axis, "C")

    # D: Label-based classification counts per 1,000 assessed participants.
    impact_axis.plot(
        clinical_impact["threshold_probability"],
        clinical_impact["high_risk_per_1000"],
        color=MODEL_COLORS["LR"],
        linewidth=1.25,
        label="Classified high risk",
    )
    impact_axis.plot(
        clinical_impact["threshold_probability"],
        clinical_impact["high_risk_with_outcome_per_1000"],
        color=CLASS_COLORS["Rapid"],
        linestyle="--",
        linewidth=1.25,
        label="High risk, assigned rapid",
    )
    impact_axis.set_xlim(0.01, 0.50)
    impact_axis.set_ylim(bottom=0)
    impact_axis.set_xlabel("Threshold probability")
    impact_axis.set_ylabel("Number per 1,000")
    impact_axis.set_title("Classification counts")
    impact_axis.grid(alpha=0.25)
    impact_axis.legend(frameon=True, ncol=1, loc="upper right")
    fs.panel_letter(impact_axis, "D")

    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def main():
    thresholds = np.linspace(0.01, 0.50, 200)
    y_true, predictions = load_predictions()
    results = build_results(y_true, predictions, thresholds)
    clinical_impact = build_clinical_impact(y_true, predictions["LR"], thresholds)
    results.to_csv(OUTPUT_CSV, index=False)
    clinical_impact.to_csv(OUTPUT_IMPACT_CSV, index=False)
    plot_results(results, clinical_impact, y_true, predictions)

    print(
        f"Loaded held-out predictions: N={len(y_true)}, "
        f"events={int(y_true.sum())}, prevalence={np.mean(y_true):.4f}"
    )
    print(f"Saved: {OUTPUT_CSV.name}")
    print(f"Saved: {OUTPUT_IMPACT_CSV.name}")
    print(f"Saved: {OUTPUT_PDF.name}")


if __name__ == "__main__":
    main()
