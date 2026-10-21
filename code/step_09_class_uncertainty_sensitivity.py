"""Sensitivity analyses that propagate LCA posterior classification uncertainty."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from lca_utils import subtype_posterior_column


CODE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_PSEUDO_REPLICATES = 200


def logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def read_model_parameters() -> dict:
    row = pd.read_csv(CODE_DIR / "step_11_lr_best_hyperparameters.csv").iloc[0]
    penalty = None if str(row["penalty"]).lower() == "none" else str(row["penalty"])
    return {
        "penalty": penalty,
        "solver": str(row["solver"]),
        "C": float(row["C"]),
        "l1_ratio": None if pd.isna(row["l1_ratio"]) else float(row["l1_ratio"]),
    }


def create_model(parameters: dict, random_state: int) -> LogisticRegression:
    kwargs = {
        "penalty": parameters["penalty"],
        "solver": parameters["solver"],
        "C": parameters["C"],
        "class_weight": "balanced",
        "max_iter": 10000,
        "random_state": random_state,
    }
    if parameters["penalty"] == "elasticnet":
        kwargs["l1_ratio"] = parameters["l1_ratio"]
    return LogisticRegression(**kwargs)


def fit_calibrated_lr(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    parameters: dict,
    random_state: int,
) -> np.ndarray:
    """Fit fold-specific preprocessing and OOF Platt calibration on training data."""
    n_splits = min(5, int(np.bincount(y_train).min()))
    if n_splits < 2:
        raise ValueError("Insufficient class counts for OOF calibration")
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_raw = np.zeros(len(y_train), dtype=float)
    for train_index, validation_index in folds.split(x_train, y_train):
        scaler = StandardScaler()
        fold_train = scaler.fit_transform(x_train.iloc[train_index])
        fold_validation = scaler.transform(x_train.iloc[validation_index])
        model = create_model(parameters, random_state)
        model.fit(fold_train, y_train[train_index])
        oof_raw[validation_index] = model.predict_proba(fold_validation)[:, 1]

    calibrator = LogisticRegression(
        penalty=None, solver="lbfgs", random_state=random_state
    )
    calibrator.fit(logit(oof_raw).reshape(-1, 1), y_train)

    scaler = StandardScaler()
    full_train = scaler.fit_transform(x_train)
    transformed_test = scaler.transform(x_test)
    model = create_model(parameters, random_state)
    model.fit(full_train, y_train)
    raw_test = model.predict_proba(transformed_test)[:, 1]
    return calibrator.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1]


def hard_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "auroc": roc_auc_score(y_true, probabilities),
        "auprc": average_precision_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
    }


def main() -> None:
    selected = pd.read_csv(CODE_DIR / "step_09_selected_features.csv")[
        "feature"
    ].tolist()
    train = pd.read_csv(CODE_DIR / "PPMI_4_train.csv")
    test = pd.read_csv(CODE_DIR / "PPMI_4_test.csv")
    subtype = pd.read_csv(CODE_DIR / "PPMI_3_patients_subtypes_lca_k2.csv")
    rapid_posterior = subtype_posterior_column(subtype, "Rapid")
    subtype = subtype[["PATNO", "Subtype_Prob", rapid_posterior]]
    parameters = read_model_parameters()

    train = train.merge(subtype, on="PATNO", how="left", validate="one_to_one")
    test = test.merge(subtype, on="PATNO", how="left", validate="one_to_one")
    if train[["Subtype_Prob", rapid_posterior]].isna().any().any():
        raise ValueError("Training records are missing LCA posterior probabilities")
    if test[["Subtype_Prob", rapid_posterior]].isna().any().any():
        raise ValueError("Test records are missing LCA posterior probabilities")

    x_train = train[selected]
    x_test = test[selected]
    y_train = train["Target"].astype(int).to_numpy()
    y_test = test["Target"].astype(int).to_numpy()
    q_train = train[rapid_posterior].to_numpy(dtype=float)
    q_test = test[rapid_posterior].to_numpy(dtype=float)

    primary_predictions = pd.read_csv(CODE_DIR / "step_11_lr_predictions.csv")
    required_prediction_columns = {"PATNO", "y_true", "y_prob"}
    missing_columns = required_prediction_columns - set(primary_predictions.columns)
    if missing_columns:
        raise ValueError(
            "Primary LR predictions are missing columns: " f"{sorted(missing_columns)}"
        )
    if (
        primary_predictions["PATNO"].isna().any()
        or primary_predictions["PATNO"].duplicated().any()
    ):
        raise ValueError(
            "Primary LR predictions require unique, non-missing PATNO values"
        )

    test_ids = pd.Index(test["PATNO"], name="PATNO")
    prediction_ids = pd.Index(primary_predictions["PATNO"], name="PATNO")
    missing_patients = test_ids.difference(prediction_ids)
    extra_patients = prediction_ids.difference(test_ids)
    if len(missing_patients) or len(extra_patients):
        raise ValueError(
            "Primary LR predictions do not match the current test cohort: "
            f"missing={len(missing_patients)}, extra={len(extra_patients)}. "
            "Run step_11_lr.py before this sensitivity analysis."
        )
    primary_predictions = (
        primary_predictions.set_index("PATNO", verify_integrity=True)
        .loc[test_ids]
        .reset_index()
    )
    if not np.array_equal(primary_predictions["y_true"].astype(int).to_numpy(), y_test):
        raise ValueError("Primary LR labels disagree with the current test cohort")

    summary_rows = []
    primary_metrics = hard_metrics(
        primary_predictions["y_true"].to_numpy(),
        primary_predictions["y_prob"].to_numpy(),
    )
    summary_rows.append(
        {
            "analysis": "Primary hard labels",
            "train_n": len(train),
            "test_n": len(test),
            "test_events_or_mean_posterior": int(y_test.sum()),
            **primary_metrics,
            "brier_target": "hard label",
        }
    )

    for threshold in (0.80, 0.90):
        train_mask = train["Subtype_Prob"].to_numpy() >= threshold
        test_mask = test["Subtype_Prob"].to_numpy() >= threshold
        probabilities = fit_calibrated_lr(
            x_train.loc[train_mask].reset_index(drop=True),
            y_train[train_mask],
            x_test.loc[test_mask].reset_index(drop=True),
            parameters,
            RANDOM_STATE + int(threshold * 100),
        )
        metrics = hard_metrics(y_test[test_mask], probabilities)
        summary_rows.append(
            {
                "analysis": f"Maximum posterior >= {threshold:.2f}",
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                "test_events_or_mean_posterior": int(y_test[test_mask].sum()),
                **metrics,
                "brier_target": "hard label",
            }
        )

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(x_train)
    scaled_test = scaler.transform(x_test)
    soft_model = sm.GLM(
        q_train,
        sm.add_constant(scaled_train),
        family=sm.families.Binomial(),
    ).fit()
    soft_probabilities = np.asarray(
        soft_model.predict(sm.add_constant(scaled_test)), dtype=float
    )
    soft_hard_metrics = hard_metrics(y_test, soft_probabilities)
    summary_rows.append(
        {
            "analysis": "Posterior-weighted fractional outcome",
            "train_n": len(train),
            "test_n": len(test),
            "test_events_or_mean_posterior": float(q_test.mean()),
            **soft_hard_metrics,
            "brier": float(np.mean((q_test - soft_probabilities) ** 2)),
            "brier_target": "rapid-class posterior",
        }
    )

    rng = np.random.default_rng(RANDOM_STATE)
    replicate_rows = []
    for replicate in range(1, N_PSEUDO_REPLICATES + 1):
        pseudo_train = rng.binomial(1, q_train)
        pseudo_test = rng.binomial(1, q_test)
        probabilities = fit_calibrated_lr(
            x_train,
            pseudo_train,
            x_test,
            parameters,
            RANDOM_STATE + 1000 + replicate,
        )
        metrics = hard_metrics(pseudo_test, probabilities)
        replicate_rows.append(
            {
                "replicate": replicate,
                "train_events": int(pseudo_train.sum()),
                "test_events": int(pseudo_test.sum()),
                **metrics,
            }
        )
    replicates = pd.DataFrame(replicate_rows)
    replicates.to_csv(
        CODE_DIR / "step_09_class_uncertainty_pseudolabel_replicates.csv",
        index=False,
    )
    pseudo_summary = {
        "analysis": "Posterior pseudo-label draws (mean)",
        "train_n": len(train),
        "test_n": len(test),
        "test_events_or_mean_posterior": replicates["test_events"].mean(),
        "auroc": replicates["auroc"].mean(),
        "auprc": replicates["auprc"].mean(),
        "brier": replicates["brier"].mean(),
        "brier_target": "sampled pseudo-label",
    }
    for metric in ("auroc", "auprc", "brier"):
        pseudo_summary[f"{metric}_p025"] = replicates[metric].quantile(0.025)
        pseudo_summary[f"{metric}_p975"] = replicates[metric].quantile(0.975)
    summary_rows.append(pseudo_summary)

    summary = pd.DataFrame(summary_rows)
    for metric in ("auroc", "auprc", "brier"):
        if f"{metric}_p025" not in summary:
            summary[f"{metric}_p025"] = np.nan
            summary[f"{metric}_p975"] = np.nan
    summary.to_csv(CODE_DIR / "step_09_class_uncertainty_summary.csv", index=False)

    prediction_output = pd.DataFrame(
        {
            "PATNO": test["PATNO"],
            "hard_label": y_test,
            "rapid_class_posterior": q_test,
            "primary_lr_probability": primary_predictions.set_index("PATNO")
            .loc[test["PATNO"], "y_prob"]
            .to_numpy(),
            "posterior_weighted_lr_probability": soft_probabilities,
        }
    )
    prediction_output.to_csv(
        CODE_DIR / "step_09_class_uncertainty_predictions.csv", index=False
    )

    print(summary.to_string(index=False))
    print(
        f"Pseudo-label sensitivity: {N_PSEUDO_REPLICATES} deterministic posterior draws "
        f"with seed {RANDOM_STATE}."
    )


if __name__ == "__main__":
    main()
