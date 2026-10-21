"""Repeated nested cross-validation sensitivity analysis for the primary LR model.

Every outer training fold repeats imputation, scaling, maximum-AUPRC LASSO selection,
backward elimination, and inner hyperparameter tuning.  The held-out prediction
test set is never read by this script.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from baseline_preprocessing import fit_baseline_preprocessor


CODE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_REPEATS = 5
N_INNER_SPLITS = 5
LASSO_CS = np.logspace(-3, 1, 41)
FINAL_CS = np.logspace(-3, 1, 13)


def fit_preprocessor(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fold-specific imputation, nominal race encoding, and scaling."""
    result = fit_baseline_preprocessor(train, validation)
    return result.train_scaled, result.validation_scaled


def lasso_max_features(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    random_state: int,
) -> list[str]:
    """Select features by maximum inner-CV AUPRC with fold-specific preprocessing."""
    inner = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=random_state
    )
    fold_scores = np.zeros((N_INNER_SPLITS, len(LASSO_CS)), dtype=float)

    for fold_index, (train_index, validation_index) in enumerate(
        inner.split(frame, outcome)
    ):
        x_train, x_validation = fit_preprocessor(
            frame.iloc[train_index], frame.iloc[validation_index]
        )
        y_train = outcome[train_index]
        y_validation = outcome[validation_index]
        for c_index, c_value in enumerate(LASSO_CS):
            model = LogisticRegression(
                penalty="l1",
                C=c_value,
                solver="liblinear",
                max_iter=5000,
                random_state=random_state,
            )
            model.fit(x_train, y_train)
            fold_scores[fold_index, c_index] = average_precision_score(
                y_validation, model.predict_proba(x_validation)[:, 1]
            )

    mean_scores = fold_scores.mean(axis=0)
    best_index = int(np.argmax(mean_scores))

    x_full, _ = fit_preprocessor(frame, frame)
    selector = LogisticRegression(
        penalty="l1",
        C=float(LASSO_CS[best_index]),
        solver="liblinear",
        max_iter=5000,
        random_state=random_state,
    )
    selector.fit(x_full, outcome)
    selected = [
        column
        for column, coefficient in zip(x_full.columns, selector.coef_[0])
        if coefficient != 0
    ]
    if not selected:
        selected = [x_full.columns[int(np.argmax(np.abs(selector.coef_[0])))]]
    return selected


def backward_elimination(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    candidates: list[str],
) -> list[str]:
    """Repeat the manuscript's P<0.05 backward rule inside an outer fold."""
    x_scaled, _ = fit_preprocessor(frame, frame)
    design = x_scaled[candidates]
    selected = list(candidates)

    while len(selected) > 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = sm.Logit(outcome, sm.add_constant(design[selected])).fit(
                    disp=False, maxiter=200
                )
            p_values = fitted.pvalues.drop("const", errors="ignore")
            if p_values.empty or not np.isfinite(p_values).all():
                break
            worst = str(p_values.idxmax())
            if float(p_values.loc[worst]) < 0.05:
                break
            selected.remove(worst)
        except (np.linalg.LinAlgError, ValueError):
            break
    return selected


def tune_final_lr(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    candidates: list[str],
    random_state: int,
) -> tuple[str, float]:
    inner = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=random_state
    )
    best_score = -np.inf
    best_parameters = ("l2", 1.0)
    for penalty in ("l1", "l2"):
        for c_value in FINAL_CS:
            scores = []
            for train_index, validation_index in inner.split(frame, outcome):
                x_train, x_validation = fit_preprocessor(
                    frame.iloc[train_index], frame.iloc[validation_index]
                )
                x_train = x_train[candidates]
                x_validation = x_validation[candidates]
                model = LogisticRegression(
                    penalty=penalty,
                    C=float(c_value),
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                )
                model.fit(x_train, outcome[train_index])
                scores.append(
                    average_precision_score(
                        outcome[validation_index],
                        model.predict_proba(x_validation)[:, 1],
                    )
                )
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_parameters = (penalty, float(c_value))
    return best_parameters


def logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def fit_inner_oof_calibrator(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    candidates: list[str],
    penalty: str,
    c_value: float,
    random_state: int,
) -> LogisticRegression:
    """Fit Platt calibration from inner OOF scores without seeing the outer fold."""
    inner = StratifiedKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=random_state
    )
    oof_probabilities = np.zeros(len(frame), dtype=float)
    for train_index, validation_index in inner.split(frame, outcome):
        x_train, x_validation = fit_preprocessor(
            frame.iloc[train_index], frame.iloc[validation_index]
        )
        x_train = x_train[candidates]
        x_validation = x_validation[candidates]
        model = LogisticRegression(
            penalty=penalty,
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
        model.fit(x_train, outcome[train_index])
        oof_probabilities[validation_index] = model.predict_proba(x_validation)[:, 1]
    calibrator = LogisticRegression(
        penalty=None, solver="lbfgs", random_state=random_state
    )
    calibrator.fit(logit(oof_probabilities).reshape(-1, 1), outcome)
    return calibrator


def main() -> None:
    data = pd.read_csv(CODE_DIR / "PPMI_4_train_unprocessed.csv")
    require_columns = {"PATNO", "Target"}
    missing = sorted(require_columns - set(data.columns))
    if missing:
        raise ValueError(f"Nested-CV input is missing columns: {missing}")
    if data["PATNO"].duplicated().any():
        raise ValueError("Nested-CV input contains duplicate PATNO values")

    raw_feature_columns = [
        column for column in data.columns if column not in require_columns
    ]
    features = data[raw_feature_columns]
    feature_columns = fit_preprocessor(features, features)[0].columns.tolist()
    outcome = data["Target"].astype(int).to_numpy()
    patient_ids = data["PATNO"].to_numpy()

    outer = RepeatedStratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    prediction_rows = []
    selection_rows = []

    for outer_index, (train_index, validation_index) in enumerate(
        outer.split(features, outcome), start=1
    ):
        repeat = (outer_index - 1) // N_OUTER_SPLITS + 1
        fold = (outer_index - 1) % N_OUTER_SPLITS + 1
        fold_seed = RANDOM_STATE + outer_index
        outer_train = features.iloc[train_index]
        outer_validation = features.iloc[validation_index]
        y_train = outcome[train_index]

        lasso_features = lasso_max_features(outer_train, y_train, fold_seed)
        selected_features = backward_elimination(outer_train, y_train, lasso_features)
        penalty, c_value = tune_final_lr(
            outer_train, y_train, selected_features, fold_seed
        )
        calibrator = fit_inner_oof_calibrator(
            outer_train,
            y_train,
            selected_features,
            penalty,
            c_value,
            fold_seed,
        )
        x_train, x_validation = fit_preprocessor(outer_train, outer_validation)
        x_train = x_train[selected_features]
        x_validation = x_validation[selected_features]
        model = LogisticRegression(
            penalty=penalty,
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=fold_seed,
        )
        model.fit(x_train, y_train)
        raw_probabilities = model.predict_proba(x_validation)[:, 1]
        probabilities = calibrator.predict_proba(
            logit(raw_probabilities).reshape(-1, 1)
        )[:, 1]

        for row_index, probability in zip(validation_index, probabilities):
            prediction_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "PATNO": patient_ids[row_index],
                    "y_true": outcome[row_index],
                    "y_prob": probability,
                    "n_lasso_features": len(lasso_features),
                    "n_final_features": len(selected_features),
                    "penalty": penalty,
                    "C": c_value,
                }
            )
        for feature in feature_columns:
            selection_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "feature": feature,
                    "lasso_selected": int(feature in lasso_features),
                    "final_selected": int(feature in selected_features),
                }
            )
        print(
            f"Nested CV repeat {repeat}/{N_REPEATS}, fold {fold}/{N_OUTER_SPLITS}: "
            f"LASSO={len(lasso_features)}, final={len(selected_features)}, "
            f"{penalty}, C={c_value:.4g}"
        )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(CODE_DIR / "step_09_nested_cv_predictions.csv", index=False)

    repeat_results = []
    for repeat, group in predictions.groupby("repeat"):
        if len(group) != len(data) or group["PATNO"].nunique() != len(data):
            raise AssertionError(
                f"Repeat {repeat} does not contain one prediction per patient"
            )
        repeat_results.append(
            {
                "repeat": repeat,
                "n": len(group),
                "events": int(group["y_true"].sum()),
                "auroc": roc_auc_score(group["y_true"], group["y_prob"]),
                "auprc": average_precision_score(group["y_true"], group["y_prob"]),
                "brier": brier_score_loss(group["y_true"], group["y_prob"]),
            }
        )
    repeat_results = pd.DataFrame(repeat_results)
    repeat_results.to_csv(CODE_DIR / "step_09_nested_cv_results.csv", index=False)

    summary_rows = []
    for metric in ("auroc", "auprc", "brier"):
        values = repeat_results[metric]
        summary_rows.append(
            {
                "metric": metric.upper(),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "minimum": values.min(),
                "maximum": values.max(),
                "n_repeats": N_REPEATS,
                "outer_folds": N_OUTER_SPLITS,
                "inner_folds": N_INNER_SPLITS,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(CODE_DIR / "step_09_nested_cv_summary.csv", index=False)

    selection = pd.DataFrame(selection_rows)
    stability = (
        selection.groupby("feature", as_index=False)[
            ["lasso_selected", "final_selected"]
        ]
        .mean()
        .rename(
            columns={
                "lasso_selected": "lasso_selection_frequency",
                "final_selected": "final_selection_frequency",
            }
        )
        .sort_values(
            ["final_selection_frequency", "lasso_selection_frequency"], ascending=False
        )
    )
    stability.to_csv(CODE_DIR / "step_09_nested_feature_stability.csv", index=False)

    print("\nRepeated nested-CV summary")
    print(summary.to_string(index=False))
    print("\nMost stable final features")
    print(stability.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
