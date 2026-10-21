"""Bootstrap 95% CIs for the four classifiers.

Reads the saved test-set predictions (y_true, y_prob) for LR, SVM, RF and
XGBoost, then reports:

  * AUROC / AUPRC / Brier point estimates with stratified bootstrap 95% CIs.
Outputs are written as CSV tables and printed to the console. This addresses
the TRIPOD+AI requirement that discrimination metrics report uncertainty.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from prediction_utils import load_aligned_prediction_frames

RANDOM_SEED = 20260702
N_BOOTSTRAP = 2000
CI_LOW, CI_HIGH = 2.5, 97.5

MODELS = {
    "LR": "step_11_lr_predictions.csv",
    "SVM": "step_11_svm_predictions.csv",
    "RF": "step_11_rf_predictions.csv",
    "XGBoost": "step_11_xgboost_predictions.csv",
}

THRESHOLD_FILES = {
    name: path.replace("_predictions.csv", "_test_performance.csv")
    for name, path in MODELS.items()
}

THRESHOLD_SECTIONS = {
    "Default threshold (0.50)": "test_default",
    "F1-optimal threshold": "test_f1",
    "Youden-optimal threshold": "test_youden",
}


def load_predictions():
    frames = load_aligned_prediction_frames(MODELS)
    return {
        name: (
            frame["y_true"].to_numpy(dtype=int),
            frame["y_prob"].to_numpy(dtype=float),
        )
        for name, frame in frames.items()
    }


def stratified_bootstrap_indices(y_true, n_boot, rng):
    """Yield resampled row indices that keep class counts fixed each draw."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    for _ in range(n_boot):
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        yield np.concatenate([boot_pos, boot_neg])


def load_thresholds():
    thresholds = {}
    for model, path in THRESHOLD_FILES.items():
        performance = pd.read_csv(path, index_col=0)
        thresholds[model] = {
            section: float(performance.loc["Optimal Threshold", column])
            for section, column in THRESHOLD_SECTIONS.items()
        }
    return thresholds


METRIC_FUNCS = {
    "AUROC": roc_auc_score,
    "AUPRC": average_precision_score,
    "Brier": brier_score_loss,
}


def logit(probabilities, eps=1e-6):
    clipped = np.clip(probabilities, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def fit_logistic_irls(y_true, x=None, offset=None, max_iter=100, tol=1e-8):
    """Fit the one- or two-parameter logistic recalibration model."""
    y_true = y_true.astype(float)
    design = (
        np.ones((len(y_true), 1), dtype=float)
        if x is None
        else np.column_stack([np.ones(len(y_true)), x.astype(float)])
    )
    offset = (
        np.zeros(len(y_true), dtype=float) if offset is None else offset.astype(float)
    )
    coefficients = np.zeros(design.shape[1], dtype=float)
    ridge = 1e-8 * np.eye(design.shape[1])

    for _ in range(max_iter):
        linear_predictor = offset + design @ coefficients
        fitted = 1 / (1 + np.exp(-np.clip(linear_predictor, -35, 35)))
        weights = np.clip(fitted * (1 - fitted), 1e-8, None)
        working_response = linear_predictor + (y_true - fitted) / weights
        lhs = design.T @ (weights[:, None] * design) + ridge
        rhs = design.T @ (weights * (working_response - offset))
        updated = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(updated - coefficients)) < tol:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def calibration_metrics(y_true, y_prob):
    linear_predictor = logit(y_prob)
    intercept = float(fit_logistic_irls(y_true, offset=linear_predictor)[0])
    slope = float(fit_logistic_irls(y_true, x=linear_predictor)[1])
    return {
        "Calibration intercept": intercept,
        "Calibration slope": slope,
    }


def threshold_metrics(y_true, y_prob, threshold):
    predicted = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "Balanced accuracy": balanced_accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "NPV": npv,
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": specificity,
        "F1-score": f1_score(y_true, predicted, zero_division=0),
        "Cohen's Kappa": cohen_kappa_score(y_true, predicted),
    }


def percentile_interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan, np.nan
    return tuple(np.percentile(finite, [CI_LOW, CI_HIGH]))


def build_full_performance_table(data, thresholds):
    """Return long-form estimates and bootstrap CIs for Supplementary Table S3."""
    rows = []
    independent_metrics = {
        "AUROC": roc_auc_score,
        "AUPRC": average_precision_score,
        "Brier score": brier_score_loss,
    }

    for model_index, (model, (y_true, y_prob)) in enumerate(data.items()):
        point_by_section = {
            "Threshold-independent and calibration": {
                **{
                    name: fn(y_true, y_prob) for name, fn in independent_metrics.items()
                },
                **calibration_metrics(y_true, y_prob),
            }
        }
        for section, threshold in thresholds[model].items():
            point_by_section[section] = {
                "Threshold": threshold,
                **threshold_metrics(y_true, y_prob, threshold),
            }

        boot_values = {
            section: {metric: [] for metric in metrics if metric != "Threshold"}
            for section, metrics in point_by_section.items()
        }
        rng = np.random.default_rng(RANDOM_SEED + model_index)
        for indices in stratified_bootstrap_indices(y_true, N_BOOTSTRAP, rng):
            boot_true = y_true[indices]
            boot_prob = y_prob[indices]
            independent = {
                **{
                    name: fn(boot_true, boot_prob)
                    for name, fn in independent_metrics.items()
                },
                **calibration_metrics(boot_true, boot_prob),
            }
            for metric, value in independent.items():
                boot_values["Threshold-independent and calibration"][metric].append(
                    value
                )
            for section, threshold in thresholds[model].items():
                for metric, value in threshold_metrics(
                    boot_true, boot_prob, threshold
                ).items():
                    boot_values[section][metric].append(value)

        for section, metrics in point_by_section.items():
            for metric, point in metrics.items():
                if metric == "Threshold":
                    ci_low, ci_high = np.nan, np.nan
                else:
                    ci_low, ci_high = percentile_interval(boot_values[section][metric])
                rows.append(
                    {
                        "Model": model,
                        "Section": section,
                        "Metric": metric,
                        "Point": point,
                        "CI_low": ci_low,
                        "CI_high": ci_high,
                    }
                )
    return pd.DataFrame(rows)


def write_supplement_latex_rows(full_table):
    """Write table rows consumed directly by supplementary_materials.tex."""
    models = list(MODELS)
    sections = [
        (
            "Threshold-independent and calibration",
            [
                "AUROC",
                "AUPRC",
                "Brier score",
                "Calibration intercept",
                "Calibration slope",
            ],
        ),
        (
            "Default threshold (0.50)",
            [
                "Accuracy",
                "Balanced accuracy",
                "Precision",
                "NPV",
                "Sensitivity",
                "Specificity",
                "F1-score",
                "Cohen's Kappa",
            ],
        ),
        (
            "F1-optimal threshold",
            [
                "Threshold",
                "Accuracy",
                "Balanced accuracy",
                "Precision",
                "NPV",
                "Sensitivity",
                "Specificity",
                "F1-score",
                "Cohen's Kappa",
            ],
        ),
        (
            "Youden-optimal threshold",
            [
                "Threshold",
                "Accuracy",
                "Balanced accuracy",
                "Precision",
                "NPV",
                "Sensitivity",
                "Specificity",
                "F1-score",
                "Cohen's Kappa",
            ],
        ),
    ]

    lines = []
    for section, metrics in sections:
        lines.append(
            rf"\multicolumn{{5}}{{@{{}}l@{{}}}}{{"
            rf"\cellcolor{{gray!15}}\textbf{{{section}}}}} \\"
        )
        for metric in metrics:
            cells = []
            for model in models:
                row = full_table.loc[
                    full_table["Model"].eq(model)
                    & full_table["Section"].eq(section)
                    & full_table["Metric"].eq(metric)
                ].iloc[0]
                if metric == "Threshold":
                    cell = f"{row['Point']:.4f}"
                else:
                    signed = metric == "Calibration intercept"
                    point = f"{row['Point']:+.4f}" if signed else f"{row['Point']:.4f}"
                    low = f"{row['CI_low']:.4f}"
                    high = f"{row['CI_high']:.4f}"
                    cell = f"{point} ({low}--{high})"
                cells.append(cell)
            label = metric
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    content = (
        r"\newcommand{\performanceTableRows}{%" + "\n" + "\n".join(lines) + "\n}\n"
    )
    Path("step_12_full_performance_table_rows.tex").write_text(
        content, encoding="utf-8"
    )


def single_model_ci(y_true, y_prob, n_boot, seed):
    rng = np.random.default_rng(seed)
    point = {m: fn(y_true, y_prob) for m, fn in METRIC_FUNCS.items()}
    boot = {m: [] for m in METRIC_FUNCS}
    for idx in stratified_bootstrap_indices(y_true, n_boot, rng):
        yt, yp = y_true[idx], y_prob[idx]
        for m, fn in METRIC_FUNCS.items():
            boot[m].append(fn(yt, yp))
    rows = {}
    for m in METRIC_FUNCS:
        arr = np.asarray(boot[m])
        lo, hi = np.percentile(arr, [CI_LOW, CI_HIGH])
        rows[m] = (point[m], lo, hi)
    return rows


def fmt(point, lo, hi):
    return f"{point:.3f} ({lo:.3f}-{hi:.3f})"


def main():
    data = load_predictions()
    thresholds = load_thresholds()

    # sanity check: every file must describe the same test set
    ref_true = next(iter(data.values()))[0]
    for name, (yt, _) in data.items():
        if len(yt) != len(ref_true) or int(yt.sum()) != int(ref_true.sum()):
            print(f"[warn] {name}: n={len(yt)}, positives={int(yt.sum())}")

    n_total = len(ref_true)
    n_pos = int(ref_true.sum())
    print("=" * 70)
    print(f"Test set: n={n_total}, positives={n_pos}, negatives={n_total - n_pos}")
    print(f"Bootstrap replicates: {N_BOOTSTRAP} (stratified, seed={RANDOM_SEED})")
    print("=" * 70)

    # ---- per-model metrics with 95% CI ----
    metric_rows = []
    for i, (name, (yt, yp)) in enumerate(data.items()):
        ci = single_model_ci(yt, yp, N_BOOTSTRAP, RANDOM_SEED + i)
        row = {"Model": name}
        print(f"\n[{name}]")
        for m in METRIC_FUNCS:
            point, lo, hi = ci[m]
            row[f"{m}"] = round(point, 4)
            row[f"{m}_CI_low"] = round(lo, 4)
            row[f"{m}_CI_high"] = round(hi, 4)
            print(f"  {m:<6}: {fmt(point, lo, hi)}")
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv("step_12_model_metrics_bootstrap_ci.csv", index=False)

    full_table = build_full_performance_table(data, thresholds)
    full_table.to_csv("step_12_full_performance_bootstrap_ci.csv", index=False)
    write_supplement_latex_rows(full_table)

    print("\nSaved:")
    print("  step_12_model_metrics_bootstrap_ci.csv")
    print("  step_12_full_performance_bootstrap_ci.csv")
    print("  step_12_full_performance_table_rows.tex")


if __name__ == "__main__":
    main()
