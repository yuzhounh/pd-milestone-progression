"""Validate cross-step consistency of the regenerated analysis outputs.

Run from the ``code`` directory after ``run_all.py``.  The checks intentionally
focus on identities that must hold across files rather than on hard-coded model
performance values, so they remain useful after legitimate data updates.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lca_utils import subtype_posterior_column


ROOT = Path(__file__).resolve().parent
EVENT_COLUMNS = [
    "Time_Walking",
    "Time_MotorComp",
    "Time_Cognition",
    "Time_Autonomic",
    "Time_Functional",
    "Time_ADL",
]
CENSOR_COLUMNS = [column.replace("Time_", "Censor_", 1) for column in EVENT_COLUMNS]
MODELS = {"LR", "SVM", "RF", "XGBoost"}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def normalized_ids(frame: pd.DataFrame) -> set[int]:
    require(frame["PATNO"].notna().all(), "PATNO contains missing values")
    return set(pd.to_numeric(frame["PATNO"], errors="raise").astype(int))


def main() -> None:
    checks: list[tuple[str, str]] = []

    annual = read_csv("PPMI_2_time_to_event.csv")
    continuous = read_csv("PPMI_2_time_to_event_continuous.csv")
    subtype = read_csv("PPMI_3_patients_subtypes_lca_k2.csv")
    require(len(annual) == len(continuous) == len(subtype), "cohort sizes disagree")
    require(
        normalized_ids(annual) == normalized_ids(continuous) == normalized_ids(subtype),
        "cohort PATNO sets disagree",
    )
    require(
        not annual["PATNO"].duplicated().any(),
        "annual time-to-event PATNO is duplicated",
    )
    require(
        not continuous["PATNO"].duplicated().any(),
        "continuous time-to-event PATNO is duplicated",
    )
    require(not subtype["PATNO"].duplicated().any(), "subtype PATNO is duplicated")

    annual_i = annual.set_index("PATNO").sort_index()
    continuous_i = continuous.set_index("PATNO").sort_index()
    require(
        np.allclose(annual_i["max_followup"], continuous_i["max_followup"]),
        "max_followup differs between annual and continuous files",
    )
    require(
        set(CENSOR_COLUMNS) <= set(annual_i.columns),
        "annual file lacks domain-specific censoring columns",
    )
    require(
        set(CENSOR_COLUMNS) <= set(continuous_i.columns),
        "continuous file lacks domain-specific censoring columns",
    )
    for column, censor_column in zip(EVENT_COLUMNS, CENSOR_COLUMNS):
        expected = np.where(
            continuous_i[column].to_numpy() == 7,
            7,
            np.ceil(continuous_i[column].to_numpy()),
        ).astype(int)
        require(
            np.array_equal(annual_i[column].to_numpy(), expected),
            f"{column} annual intervals do not equal ceil(continuous time)",
        )
        require(
            annual_i[column].between(1, 7).all(),
            f"{column} contains an invalid annual interval",
        )
        require(
            np.allclose(annual_i[censor_column], continuous_i[censor_column]),
            f"{censor_column} differs between annual and continuous files",
        )
        require(
            annual_i[censor_column].between(0, 6).all(),
            f"{censor_column} leaves the six-year window",
        )
        event = continuous_i[column].lt(7)
        require(
            np.allclose(
                continuous_i.loc[event, column], continuous_i.loc[event, censor_column]
            ),
            f"{column} event times disagree with {censor_column}",
        )
    checks.append(("cohort_and_event_time_alignment", f"PASS (N={len(annual)})"))

    subtype_counts = subtype["Subtype"].value_counts()
    require(set(subtype_counts.index) == {"Slow", "Rapid"}, "unexpected subtype labels")
    class_map = subtype.groupby("Subtype_Label")["Subtype"].agg(
        lambda values: set(values)
    )
    require(set(class_map.index) == {0, 1}, "unexpected two-class numeric labels")
    require(
        all(len(values) == 1 for values in class_map),
        "a numeric class label maps to multiple subtype names",
    )
    rapid_posterior = subtype_posterior_column(subtype, "Rapid")
    slow_posterior = subtype_posterior_column(subtype, "Slow")
    require(
        np.allclose(subtype[rapid_posterior] + subtype[slow_posterior], 1.0, atol=1e-8),
        "posterior probabilities do not sum to one",
    )
    assigned_posterior = np.where(
        subtype["Subtype"].eq("Rapid"),
        subtype[rapid_posterior],
        subtype[slow_posterior],
    )
    require(
        np.allclose(subtype["Subtype_Prob"], assigned_posterior, atol=1e-8),
        "Subtype_Prob is not the assigned class posterior",
    )
    checks.append(
        (
            "subtype_labels",
            f"PASS (Slow={subtype_counts['Slow']}; Rapid={subtype_counts['Rapid']})",
        )
    )

    train = read_csv("PPMI_4_train.csv")
    test = read_csv("PPMI_4_test.csv")
    train_ids, test_ids = normalized_ids(train), normalized_ids(test)
    require(train_ids.isdisjoint(test_ids), "training and test PATNO overlap")
    require(
        train_ids | test_ids == normalized_ids(subtype),
        "training/test union does not equal the LCA cohort",
    )
    label_map = subtype.set_index("PATNO")["Subtype"].eq("Rapid").astype(int)
    for split_name, frame in (("train", train), ("test", test)):
        require(
            (
                frame.set_index("PATNO")["Target"]
                == label_map.loc[frame["PATNO"]].to_numpy()
            ).all(),
            f"{split_name} Target disagrees with subtype label",
        )
        require("duration_years" in frame.columns, f"{split_name} lacks duration_years")
        require(
            "duration" not in frame.columns and "duration_yrs" not in frame.columns,
            f"{split_name} contains an ambiguous legacy duration column",
        )
        require(
            frame["duration_years"].notna().all(),
            f"{split_name} duration_years contains missing values",
        )
        require(
            frame["duration_years"].between(0, 2).all(),
            f"{split_name} duration_years violates the early-PD criterion",
        )
    checks.append(("prediction_split", f"PASS (train={len(train)}; test={len(test)})"))

    baseline = read_csv("step_08_baseline_source.csv")
    require(len(baseline) == len(subtype), "baseline-source audit size is wrong")
    require(
        normalized_ids(baseline) == normalized_ids(subtype),
        "baseline-source audit PATNO set is wrong",
    )
    require(
        (baseline["baseline_source_event"] == "BL").all(),
        "a non-BL record was used as baseline",
    )
    require(
        set(baseline["split"]) == {"train", "test"}, "baseline split labels are invalid"
    )
    checks.append(
        ("baseline_source_and_duration", "PASS (all BL; duration_years <= 2)")
    )

    lasso = read_csv("step_08_selected_features.csv")
    final_features = read_csv("step_09_selected_features.csv")
    lasso_features = set(lasso["feature"])
    selected_features = set(final_features["feature"])
    require(
        selected_features <= lasso_features,
        "backward-elimination features are not a subset of MAX LASSO features",
    )
    require(len(selected_features) > 0, "no final predictors were selected")
    checks.append(
        ("feature_selection", f"PASS ({len(selected_features)} final predictors)")
    )

    unprocessed_train = read_csv("PPMI_4_train_unprocessed.csv")
    require(
        normalized_ids(unprocessed_train) == train_ids,
        "unprocessed training PATNO set disagrees with the saved training split",
    )
    unprocessed_test = read_csv("PPMI_4_test_unprocessed.csv")
    race_indicators = ["race_Black", "race_Asian", "race_Other"]
    require(
        "race" in unprocessed_train and "race" in unprocessed_test,
        "unprocessed split must retain the raw nominal race code",
    )
    require(
        "race" not in train and "race" not in test,
        "processed model data must not retain ordinal race codes",
    )
    require(
        set(race_indicators) <= set(train.columns)
        and set(race_indicators) <= set(test.columns),
        "processed model data lack the fixed race indicators",
    )
    require(
        train[race_indicators].sum(axis=1).between(0, 1).all(),
        "training race indicators are not reference-coded",
    )
    require(
        test[race_indicators].sum(axis=1).between(0, 1).all(),
        "test race indicators are not reference-coded",
    )
    race_audit = read_csv("step_08_race_encoding.csv")
    require(
        set(race_audit["source_label"]) == {"White", "Black", "Asian", "Other"},
        "race encoding audit lacks an expected category",
    )
    nested_predictions = read_csv("step_09_nested_cv_predictions.csv")
    nested_summary = read_csv("step_09_nested_cv_summary.csv")
    nested_repeats = int(nested_summary["n_repeats"].iloc[0])
    require(
        set(nested_summary["metric"]) == {"AUROC", "AUPRC", "BRIER"},
        "nested-CV summary metrics are incomplete",
    )
    require(
        len(nested_predictions) == len(train) * nested_repeats,
        "nested-CV prediction count is wrong",
    )
    require(
        (nested_predictions.groupby("repeat")["PATNO"].nunique() == len(train)).all(),
        "a nested-CV repeat lacks one prediction per training participant",
    )
    require(
        nested_predictions["y_prob"].between(0, 1).all(),
        "nested-CV probabilities leave [0, 1]",
    )
    checks.append(
        (
            "nested_cross_validation",
            f"PASS ({nested_repeats}x5 outer folds; preprocessing/selection repeated)",
        )
    )

    nonlinearity = read_csv("step_09_nonlinearity_tests.csv")
    require(
        set(nonlinearity["feature"]) == selected_features,
        "functional-form diagnostics do not cover every final predictor",
    )
    spline_rows = nonlinearity["df"].gt(0)
    require(
        nonlinearity.loc[spline_rows, "p_nonlinearity"].between(0, 1).all(),
        "functional-form diagnostics contain invalid spline P values",
    )
    require(
        nonlinearity.loc[~spline_rows, "p_nonlinearity"].isna().all(),
        "binary predictors should not receive spline P values",
    )
    influence_summary = read_csv("step_09_influence_summary.csv")
    influence_sensitivity = read_csv("step_09_influence_sensitivity.csv")
    require(
        int(influence_summary.loc[0, "n"]) == len(train),
        "influence diagnostics sample size disagrees with training data",
    )
    require(
        set(influence_sensitivity["feature"]) == selected_features,
        "influence sensitivity does not cover every final predictor",
    )
    require(
        influence_sensitivity["coefficient_sign_consistent"].all(),
        "a predictor reverses direction after influence-point exclusion",
    )
    checks.append(
        (
            "functional_form_and_influence",
            f"PASS ({len(selected_features)} predictors; coefficient signs stable)",
        )
    )

    uncertainty_summary = read_csv("step_09_class_uncertainty_summary.csv")
    uncertainty_predictions = read_csv("step_09_class_uncertainty_predictions.csv")
    uncertainty_replicates = read_csv(
        "step_09_class_uncertainty_pseudolabel_replicates.csv"
    )
    required_uncertainty_analyses = {
        "Primary hard labels",
        "Maximum posterior >= 0.80",
        "Maximum posterior >= 0.90",
        "Posterior-weighted fractional outcome",
        "Posterior pseudo-label draws (mean)",
    }
    require(
        set(uncertainty_summary["analysis"]) == required_uncertainty_analyses,
        "class-uncertainty sensitivity analyses are incomplete",
    )
    require(
        normalized_ids(uncertainty_predictions) == test_ids,
        "class-uncertainty prediction PATNO set disagrees with the test set",
    )
    require(
        uncertainty_predictions[
            [
                "rapid_class_posterior",
                "primary_lr_probability",
                "posterior_weighted_lr_probability",
            ]
        ]
        .apply(lambda column: column.between(0, 1).all())
        .all(),
        "class-uncertainty probabilities leave [0, 1]",
    )
    require(
        len(uncertainty_replicates) == 200,
        "class-uncertainty pseudo-label analysis lacks 200 replicates",
    )
    checks.append(
        (
            "class_uncertainty_sensitivity",
            "PASS (high-confidence, posterior-weighted, and 200 pseudo-label analyses)",
        )
    )

    reference_prediction: pd.DataFrame | None = None
    for model in ("lr", "svm", "rf", "xgboost"):
        prediction = read_csv(f"step_11_{model}_predictions.csv").sort_values("PATNO")
        require(len(prediction) == len(test), f"{model} prediction count is wrong")
        require(
            normalized_ids(prediction) == test_ids,
            f"{model} prediction PATNO set is wrong",
        )
        require(
            prediction["y_prob"].between(0, 1).all(),
            f"{model} calibrated probabilities leave [0, 1]",
        )
        require(
            prediction["y_prob_raw"].between(0, 1).all(),
            f"{model} raw probabilities leave [0, 1]",
        )
        require(
            prediction["model_version"].notna().all(),
            f"{model} model version is missing",
        )
        require(
            (prediction["split"] == "test").all(),
            f"{model} includes non-test predictions",
        )
        if reference_prediction is None:
            reference_prediction = prediction[["PATNO", "y_true"]]
        else:
            require(
                prediction[["PATNO", "y_true"]]
                .reset_index(drop=True)
                .equals(reference_prediction.reset_index(drop=True)),
                f"{model} outcomes do not align with other prediction files",
            )
    require(
        int(reference_prediction["y_true"].sum()) == int(test["Target"].sum()),
        "prediction event count disagrees with the test set",
    )
    checks.append(
        (
            "prediction_files",
            f"PASS (N={len(test)}; events={int(test['Target'].sum())}; models=4)",
        )
    )

    performance = read_csv("step_12_full_performance_bootstrap_ci.csv")
    require(set(performance["Model"]) == MODELS, "performance table model set is wrong")
    interval_rows = performance["CI_low"].notna() & performance["CI_high"].notna()
    require(
        (
            performance.loc[interval_rows, "CI_low"]
            <= performance.loc[interval_rows, "Point"]
        ).all(),
        "a point estimate is below its confidence interval",
    )
    require(
        (
            performance.loc[interval_rows, "Point"]
            <= performance.loc[interval_rows, "CI_high"]
        ).all(),
        "a point estimate is above its confidence interval",
    )
    calibration = read_csv("step_19_all_models_calibration_summary.csv")
    require(set(calibration["Model"]) == MODELS, "calibration model set is wrong")
    require(
        (calibration["N"] == len(test)).all(),
        "calibration N disagrees with the test set",
    )
    require(
        (calibration["Events"] == int(test["Target"].sum())).all(),
        "calibration event count disagrees with the test set",
    )
    checks.append(("performance_and_calibration", "PASS (4 models aligned)"))

    nomogram = read_csv("step_13_lr_nomogram_validation.csv")
    require(
        len(nomogram) == 1 and bool(nomogram.loc[0, "validation_passed"]),
        "nomogram probability-equivalence validation failed",
    )
    require(
        int(nomogram.loc[0, "n_samples"]) == len(train),
        "nomogram validation sample size disagrees with training data",
    )
    checks.append(("nomogram_equivalence", "PASS"))

    for filename, feature_column in (
        ("step_15_or_results.csv", "Feature"),
        ("step_16_vif_results.csv", "Feature"),
    ):
        require(
            set(read_csv(filename)[feature_column]) == selected_features,
            f"{filename} feature set disagrees with the final model",
        )
    shap = read_csv("step_17_shap_importance.csv")
    require(set(shap["Model"]) == MODELS, "SHAP model set is wrong")
    require(
        shap.groupby("Model")["Feature"]
        .apply(set)
        .map(lambda value: value == set(read_csv("step_16_vif_results.csv")["Display"]))
        .all(),
        "SHAP feature sets disagree with the final model",
    )
    checks.append(("downstream_feature_sets", "PASS"))

    source_audit = read_csv("step_01_source_match_audit.csv")
    require(
        source_audit["union_visit_keys"].nunique() == 1,
        "source audit reports inconsistent union sizes",
    )
    vital = source_audit.loc[source_audit["source"] == "Vital Signs"]
    require(len(vital) == 1, "source audit lacks a unique Vital Signs row")
    require(
        int(vital.iloc[0]["duplicate_visit_keys_before_resolution"]) == 0,
        "Vital Signs contains duplicate visit keys",
    )
    checks.append(
        (
            "visit_key_union",
            f"PASS (union={int(source_audit['union_visit_keys'].iloc[0])}; vital duplicates=0)",
        )
    )

    summary = pd.DataFrame(checks, columns=["check", "result"])
    summary.to_csv(ROOT / "pipeline_validation_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nAll pipeline consistency checks passed.")


if __name__ == "__main__":
    main()
