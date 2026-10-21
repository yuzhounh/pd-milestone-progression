#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare P-value and BIC backward selection after LASSO MAX screening.

This is a sensitivity experiment. It reads the existing train/test split and
does not replace the production outputs from ``step_09_backward_stepwise.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


CODE_DIR = Path(__file__).resolve().parent
P_VALUE_THRESHOLD = 0.05
BIC_TOLERANCE = 1e-8


def fit_logit(data: pd.DataFrame, outcome: np.ndarray, features: list[str]):
    """Fit an unpenalized logistic model with an intercept."""
    design = sm.add_constant(data[features], has_constant="add")
    return sm.Logit(outcome, design).fit(disp=False)


def p_value_backward(
    data: pd.DataFrame, outcome: np.ndarray, candidates: list[str]
) -> tuple[list[str], list[dict]]:
    """Replicate the production Wald-P backward-elimination rule."""
    selected = list(candidates)
    history = []
    while len(selected) > 1:
        model = fit_logit(data, outcome, selected)
        p_values = model.pvalues.drop("const", errors="ignore")
        worst_feature = str(p_values.idxmax())
        worst_p = float(p_values.loc[worst_feature])
        if worst_p < P_VALUE_THRESHOLD:
            history.append(
                {
                    "method": "P-value",
                    "step": len(history) + 1,
                    "action": "stop",
                    "dropped_feature": "",
                    "criterion_before": worst_p,
                    "criterion_after": worst_p,
                    "bic": float(model.bic),
                    "n_features": len(selected),
                }
            )
            break
        selected.remove(worst_feature)
        reduced_model = fit_logit(data, outcome, selected)
        history.append(
            {
                "method": "P-value",
                "step": len(history) + 1,
                "action": "drop",
                "dropped_feature": worst_feature,
                "criterion_before": worst_p,
                "criterion_after": float(
                    reduced_model.pvalues.drop("const", errors="ignore").max()
                ),
                "bic": float(reduced_model.bic),
                "n_features": len(selected),
            }
        )
    return selected, history


def bic_backward(
    data: pd.DataFrame, outcome: np.ndarray, candidates: list[str]
) -> tuple[list[str], list[dict]]:
    """Drop the single variable giving the largest BIC improvement each step."""
    selected = list(candidates)
    history = []
    while len(selected) > 1:
        current_model = fit_logit(data, outcome, selected)
        current_bic = float(current_model.bic)
        trials = []
        for feature in selected:
            trial_features = [item for item in selected if item != feature]
            trial_model = fit_logit(data, outcome, trial_features)
            trials.append((float(trial_model.bic), feature))
        best_bic, best_feature = min(trials)
        improves = best_bic < current_bic - BIC_TOLERANCE
        history.append(
            {
                "method": "BIC",
                "step": len(history) + 1,
                "action": "drop" if improves else "stop",
                "dropped_feature": best_feature if improves else "",
                "criterion_before": current_bic,
                "criterion_after": best_bic,
                "bic": best_bic if improves else current_bic,
                "n_features": len(selected) - 1 if improves else len(selected),
            }
        )
        if not improves:
            break
        selected.remove(best_feature)
    return selected, history


def performance_rows(
    method: str,
    features: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict]:
    """Evaluate one selected model on train and untouched test data."""
    model = fit_logit(train, train["Target"].to_numpy(), features)
    rows = []
    for split, data in (("train", train), ("test", test)):
        outcome = data["Target"].to_numpy()
        design = sm.add_constant(data[features], has_constant="add")
        probability = np.asarray(model.predict(design))
        rows.append(
            {
                "method": method,
                "split": split,
                "n_features": len(features),
                "features": ";".join(features),
                "train_bic": float(model.bic),
                "auroc": roc_auc_score(outcome, probability),
                "auprc": average_precision_score(outcome, probability),
                "brier": brier_score_loss(outcome, probability),
                "log_loss": log_loss(outcome, probability),
            }
        )
    return rows


def paired_bootstrap_differences(
    train: pd.DataFrame,
    test: pd.DataFrame,
    p_features: list[str],
    bic_features: list[str],
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Estimate paired test-set metric differences (BIC minus P-value)."""
    outcome = test["Target"].to_numpy()
    probabilities = {}
    for method, features in (("P-value", p_features), ("BIC", bic_features)):
        model = fit_logit(train, train["Target"].to_numpy(), features)
        design = sm.add_constant(test[features], has_constant="add")
        probabilities[method] = np.asarray(model.predict(design))

    metric_functions = {
        "auroc": roc_auc_score,
        "auprc": average_precision_score,
        "brier": brier_score_loss,
        "log_loss": log_loss,
    }
    point_differences = {
        metric: function(outcome, probabilities["BIC"])
        - function(outcome, probabilities["P-value"])
        for metric, function in metric_functions.items()
    }

    rng = np.random.default_rng(20260927)
    bootstrap = {metric: [] for metric in metric_functions}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(test), len(test))
        sampled_outcome = outcome[indices]
        if np.unique(sampled_outcome).size < 2:
            continue
        for metric, function in metric_functions.items():
            bootstrap[metric].append(
                function(sampled_outcome, probabilities["BIC"][indices])
                - function(sampled_outcome, probabilities["P-value"][indices])
            )

    rows = []
    for metric, differences in bootstrap.items():
        rows.append(
            {
                "metric": metric,
                "difference_definition": "BIC minus P-value",
                "point_difference": point_differences[metric],
                "bootstrap_ci_lower": np.quantile(differences, 0.025),
                "bootstrap_ci_upper": np.quantile(differences, 0.975),
                "n_bootstrap": len(differences),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train = pd.read_csv(CODE_DIR / "PPMI_4_train.csv")
    test = pd.read_csv(CODE_DIR / "PPMI_4_test.csv")
    candidates = pd.read_csv(CODE_DIR / "step_08_selected_features.csv")[
        "feature"
    ].tolist()

    p_features, p_history = p_value_backward(
        train, train["Target"].to_numpy(), candidates
    )
    bic_features, bic_history = bic_backward(
        train, train["Target"].to_numpy(), candidates
    )

    selection_rows = []
    for feature in candidates:
        selection_rows.append(
            {
                "feature": feature,
                "selected_by_p_value": int(feature in p_features),
                "selected_by_bic": int(feature in bic_features),
            }
        )
    pd.DataFrame(selection_rows).to_csv(
        CODE_DIR / "step_09_bic_selection_comparison.csv", index=False
    )
    pd.DataFrame(p_history + bic_history).to_csv(
        CODE_DIR / "step_09_bic_stepwise_history.csv", index=False
    )

    metrics = []
    metrics.extend(performance_rows("P-value", p_features, train, test))
    metrics.extend(performance_rows("BIC", bic_features, train, test))
    pd.DataFrame(metrics).to_csv(
        CODE_DIR / "step_09_bic_performance_comparison.csv", index=False
    )
    bootstrap_differences = paired_bootstrap_differences(
        train, test, p_features, bic_features
    )
    bootstrap_differences.to_csv(
        CODE_DIR / "step_09_bic_bootstrap_differences.csv", index=False
    )

    print(f"LASSO candidates: {len(candidates)}")
    print(f"P-value selected ({len(p_features)}): {p_features}")
    print(f"BIC selected ({len(bic_features)}): {bic_features}")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(bootstrap_differences.to_string(index=False))


if __name__ == "__main__":
    main()
