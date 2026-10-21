"""Step 19: calibration intercept and slope of the four classifiers on the test set."""

import numpy as np
import pandas as pd
from typing import Optional

from prediction_utils import load_aligned_prediction_frames


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic_irls(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    y = y.astype(float)
    if x is None:
        design = np.ones((len(y), 1), dtype=float)
    else:
        design = np.column_stack([np.ones(len(y)), x.astype(float)])

    if offset is None:
        offset = np.zeros(len(y), dtype=float)
    else:
        offset = offset.astype(float)

    coef = np.zeros(design.shape[1], dtype=float)
    ridge = 1e-8 * np.eye(design.shape[1])

    for _ in range(max_iter):
        eta = offset + design @ coef
        p = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        w = np.clip(p * (1 - p), 1e-8, None)
        z = eta + (y - p) / w

        lhs = design.T @ (w[:, None] * design) + ridge
        rhs = design.T @ (w * (z - offset))
        new_coef = np.linalg.solve(lhs, rhs)

        if np.max(np.abs(new_coef - coef)) < tol:
            coef = new_coef
            break
        coef = new_coef

    return coef


def compute_calibration_for_model(model_name: str, pred: pd.DataFrame) -> dict:
    y = pred["y_true"].astype(int).to_numpy()
    p = pred["y_prob"].astype(float).to_numpy()
    lp = logit(p)

    brier = float(np.mean((p - y) ** 2))
    intercept_offset = float(fit_logistic_irls(y, offset=lp)[0])
    recalibration_coef = fit_logistic_irls(y, x=lp)
    recalibration_intercept = float(recalibration_coef[0])
    calibration_slope = float(recalibration_coef[1])

    # Expected Calibration Error (ECE) with 10 bins
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        bin_mask = (p >= bin_boundaries[i]) & (p < bin_boundaries[i + 1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y[bin_mask])
            bin_conf = np.mean(p[bin_mask])
            ece += (np.sum(bin_mask) / len(p)) * np.abs(bin_acc - bin_conf)

    return {
        "Model": model_name,
        "N": len(y),
        "Events": int(y.sum()),
        "Observed_event_rate": float(y.mean()),
        "Mean_predicted_risk": float(p.mean()),
        "Brier_score": brier,
        "Calibration_intercept": intercept_offset,
        "Recalibration_intercept": recalibration_intercept,
        "Calibration_slope": calibration_slope,
        "ECE_10bins": ece,
    }


def main():
    models = {
        "LR": "step_11_lr_predictions.csv",
        "SVM": "step_11_svm_predictions.csv",
        "RF": "step_11_rf_predictions.csv",
        "XGBoost": "step_11_xgboost_predictions.csv",
    }
    prediction_frames = load_aligned_prediction_frames(models)
    rows = [
        compute_calibration_for_model(name, prediction_frames[name]) for name in models
    ]

    df = pd.DataFrame(rows)
    df.to_csv(
        "step_19_all_models_calibration_summary.csv", index=False, encoding="utf-8-sig"
    )
    print("\nAll Models Calibration Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print("Saved: step_19_all_models_calibration_summary.csv")


if __name__ == "__main__":
    main()
