"""Sensitivity of the locked LR to removing the unstable predictors OH (and UPSIT).

Exploratory, post hoc: the test set was already used to evaluate the locked eight-predictor model.
The procedure repeats step_11_lr.py in memory without writing any of its outputs: the locked
hyperparameters (step_11_lr_best_hyperparameters.csv; no retuning), class weighting, training-set
standardization, 10-fold OOF Platt calibration and refit on the full training set. Test-set AUROC,
AUPRC and Brier score use 2,000 class-stratified bootstrap resamples (as in step_12); differences from
the eight-predictor model are paired on the same resamples. The eight-predictor refit must reproduce
the saved calibrated test probabilities (step_11_lr_predictions.csv). Output:
step_21_lr_reduced_sensitivity.csv.
"""

import importlib.util

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from utils import load_data, prepare_data, stratified_cross_validation

spec = importlib.util.spec_from_file_location("lr", "step_11_lr.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)

N_BOOT, SEED = 2000, 20260702
METRICS = {
    "AUROC": roc_auc_score,
    "AUPRC": average_precision_score,
    "Brier": brier_score_loss,
}
SETS = {
    "8 predictors (locked)": [],
    "without OH": ["orthostasis"],
    "without OH and UPSIT": ["orthostasis", "upsit_pctl"],
}


def calibrated_test_probabilities(model_factory, X_tr, y_tr, X_te):
    oof = np.zeros(len(y_tr))
    for tr, va in stratified_cross_validation(X_tr, y_tr, n_folds=10, random_state=42):
        m = model_factory()
        m.fit(X_tr[tr], y_tr[tr])
        oof[va] = m.predict_proba(X_tr[va])[:, 1]
    calibrator = LogisticRegression(penalty=None, solver="lbfgs", random_state=42)
    calibrator.fit(lr.logit(oof).reshape(-1, 1), y_tr)
    final = model_factory()
    final.fit(X_tr, y_tr)
    raw = final.predict_proba(X_te)[:, 1]
    return calibrator.predict_proba(lr.logit(raw).reshape(-1, 1))[:, 1]


def main():
    train, test, _, _ = load_data(
        "PPMI_4_train.csv",
        "PPMI_4_test.csv",
        "PPMI_feature_mapping.csv",
        "step_09_selected_features.csv",
        return_scaler=True,
    )
    Xtr_df, y_tr, _ = prepare_data(train)
    Xte_df, y_te, _ = prepare_data(test)
    y_tr, y_te = y_tr.to_numpy(), y_te.to_numpy()
    params = pd.read_csv(
        "step_11_lr_best_hyperparameters.csv", encoding="utf-8-sig"
    ).iloc[0]
    l1 = (
        None
        if pd.isna(params["l1_ratio"]) or params["l1_ratio"] == ""
        else float(params["l1_ratio"])
    )
    model = lr.LRModel(random_state=42)
    factory = lambda: model.create_model(
        [params["penalty"], params["solver"], float(params["C"]), l1]
    )

    probs = {}
    for name, drop in SETS.items():
        cols = [c for c in Xtr_df.columns if c not in drop]
        probs[name] = calibrated_test_probabilities(
            factory, Xtr_df[cols].to_numpy(), y_tr, Xte_df[cols].to_numpy()
        )
    saved = pd.read_csv("step_11_lr_predictions.csv")
    prob_col = [
        c
        for c in saved.columns
        if "cal" in c.lower() or c.lower() in ("y_prob", "prob")
    ][0]
    diff = float(
        np.max(np.abs(saved[prob_col].to_numpy() - probs["8 predictors (locked)"]))
    )
    if diff > 1e-8:
        raise SystemExit(
            f"Eight-predictor refit does not reproduce saved probabilities: {diff}"
        )

    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
    draws = [
        np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for _ in range(N_BOOT)
    ]
    base = probs["8 predictors (locked)"]
    rows = []
    for name, p in probs.items():
        for metric, fn in METRICS.items():
            point = fn(y_te, p)
            boot = np.array([fn(y_te[i], p[i]) for i in draws])
            dpoint = point - fn(y_te, base)
            dboot = np.array([fn(y_te[i], p[i]) - fn(y_te[i], base[i]) for i in draws])
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": point,
                    "ci_low": np.percentile(boot, 2.5),
                    "ci_high": np.percentile(boot, 97.5),
                    "diff_vs_locked": dpoint,
                    "diff_ci_low": np.percentile(dboot, 2.5),
                    "diff_ci_high": np.percentile(dboot, 97.5),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv("step_21_lr_reduced_sensitivity.csv", index=False)
    print("reproduction max |diff|:", diff)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
