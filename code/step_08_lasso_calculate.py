"""
Step 08: training-only preprocessing and LASSO screening of candidate predictors.

Tasks:
1. Load the patient subtypes and baseline clinical features
2. Map and construct features (categorical variables; duration in months and years)
3. Preprocess (missingness filter, imputation, standardisation)
4. LASSO cross-validation (penalty with the highest mean AUC-PR)
5. Save the datasets and feature sets as CSV files

Outputs:
- PPMI_4_train.csv / PPMI_4_test.csv: all features on the original scale (race as reference-coded indicators)
- step_08_baseline_source.csv: baseline source visit and split of each participant
- step_08_candidate_features_cleaned.csv: full list of encoded candidate features
- step_08_cv_scores.csv: cross-validation score curve
- step_08_cv_best_params.csv: best parameters
- step_08_coef_path.csv: coefficient paths
- step_08_selected_features.csv: features selected by the maximum mean CV AUPRC rule
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from baseline_preprocessing import fit_baseline_preprocessor, race_encoding_audit
from baseline_utils import (
    BASELINE_SOURCE_COLUMN,
    CURATED_FILENAME,
    baseline_correction_audit,
    load_curated_data,
    prepare_baseline_records,
)
from moca_utils import RAW_MOCA_FILENAME, moca_source_audit

# Plot font and minus sign
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = True


def main():
    """Run the preprocessing and LASSO screening and save all outputs."""
    print(">>> Running LASSO...")

    # ==========================================
    # 1. Load and merge the data
    # ==========================================
    df_subtypes = pd.read_csv("PPMI_3_patients_subtypes_lca_k2.csv")
    df_subtypes["Target"] = df_subtypes["Subtype"].apply(
        lambda x: 1 if x == "Rapid" else 0
    )

    curated_path = CURATED_FILENAME
    df_curated = load_curated_data(curated_path)

    raw_moca = pd.read_csv("tables/" + RAW_MOCA_FILENAME, low_memory=False)
    df_features = prepare_baseline_records(
        df_curated, source_path=curated_path, raw_moca=raw_moca
    )
    baseline_correction_audit(df_features).to_csv(
        "step_08_baseline_corrections.csv", index=False
    )
    moca_source_audit(df_features).to_csv("step_08_moca_source_audit.csv", index=False)

    df_full = pd.merge(
        df_subtypes[["PATNO", "Target"]], df_features, on="PATNO", how="inner"
    )
    print(f"1. Data loaded; merged sample size N={len(df_full)}")

    # ==========================================
    # 2. Map and construct features
    # ==========================================
    print("\n>>> Mapping and constructing features...")

    # fampd_bin: binary family history
    if "fampd_bin" in df_full.columns:
        df_full["fampd_bin"] = df_full["fampd_bin"].map({1: 1, 2: 0})
        print("  [Map] fampd_bin: {1, 2} --> {1, 0}")

    # handed: handedness as dummy variables
    if "handed" in df_full.columns:
        df_full["handed_left"] = df_full["handed"].map({1: 1, 2: 0, 3: 1})
        df_full["handed_right"] = df_full["handed"].map({1: 0, 2: 1, 3: 1})
        print("  [Map] handed: {1, 2, 3} --> handed_left, handed_right")

    # DOMSIDE: dominant symptom side as dummy variables
    if "DOMSIDE" in df_full.columns:
        df_full["DOMSIDE_left"] = df_full["DOMSIDE"].map({1: 1, 2: 0, 3: 1})
        df_full["DOMSIDE_right"] = df_full["DOMSIDE"].map({1: 0, 2: 1, 3: 1})
        print("  [Map] DOMSIDE: {1, 2, 3} --> DOMSIDE_left, DOMSIDE_right")

    # Curated duration is in months; duration_yrs is the disease duration since diagnosis (years) used for eligibility and modelling.
    required_duration_cols = {"duration", "duration_yrs"}
    missing_duration_cols = sorted(required_duration_cols - set(df_full.columns))
    if missing_duration_cols:
        raise ValueError(
            f"Baseline data lack duration columns: {missing_duration_cols}"
        )
    df_full["duration_months"] = pd.to_numeric(df_full["duration"], errors="coerce")
    df_full["duration_years"] = pd.to_numeric(df_full["duration_yrs"], errors="coerce")
    duration_consistency_error = (
        df_full["duration_months"] / 12 - df_full["duration_years"]
    ).abs()
    if duration_consistency_error.dropna().gt(0.01).any():
        raise ValueError("duration (months) and duration_yrs (years) disagree")
    if df_full["duration_years"].isna().any():
        raise ValueError("The analysis cohort has missing baseline disease duration")
    if df_full["duration_years"].gt(2).any():
        raise ValueError(
            "The analysis cohort has patients with baseline disease duration > 2 years"
        )
    print(
        "  [Construct] duration_months and duration_years (baseline disease duration)"
    )

    # ==========================================
    # 3. Load the candidate-feature list
    # ==========================================
    feature_flags_df = pd.read_csv("PPMI_initial_feature_sets.csv")
    perfect_features = feature_flags_df[feature_flags_df["Flag"] == 1][
        "Feature"
    ].tolist()
    duration_aliases = {"duration": "duration_years", "duration_yrs": "duration_years"}
    perfect_features = [
        duration_aliases.get(feature, feature) for feature in perfect_features
    ]
    perfect_features = list(dict.fromkeys(perfect_features))
    exclude_cols = [
        "PATNO",
        "COHORT",
        "SITE",
        "Target",
        "Subtype",
        "EVENT_ID",
        "YEAR",
        "visit_date",
    ]
    perfect_features = [
        f
        for f in perfect_features
        if f not in exclude_cols and not f.startswith("moca_")
    ]

    # Replace variables by their dummy columns
    if "handed" in perfect_features:
        perfect_features.remove("handed")
        perfect_features.extend(["handed_left", "handed_right"])
    if "DOMSIDE" in perfect_features:
        perfect_features.remove("DOMSIDE")
        perfect_features.extend(["DOMSIDE_left", "DOMSIDE_right"])
    print(
        f"  [Info] {len(perfect_features)} candidate features after processing: {perfect_features}"
    )

    # Check that the features exist
    valid_features = [f for f in perfect_features if f in df_full.columns]
    missing_features = [f for f in perfect_features if f not in df_full.columns]
    if missing_features:
        print(f"  [Info] Features not found and ignored: {missing_features}")
    print(f"  [Valid features] {len(valid_features)}: {valid_features}")

    X_all = df_full[valid_features].copy()
    y = df_full["Target"].values
    patno_series = df_full["PATNO"].values
    baseline_source_series = df_full[BASELINE_SOURCE_COLUMN].values

    # ==========================================
    # 4. Split and preprocess
    # ==========================================
    X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
        X_all, y, range(len(df_full)), test_size=0.3, random_state=42, stratify=y
    )
    train_patno = patno_series[list(train_idx)]
    test_patno = patno_series[list(test_idx)]
    train_source = baseline_source_series[list(train_idx)]
    test_source = baseline_source_series[list(test_idx)]
    print(f"2. Data split: training {len(X_train_raw)}, test {len(X_test_raw)}")

    baseline_source_audit = pd.concat(
        [
            pd.DataFrame(
                {
                    "PATNO": train_patno,
                    BASELINE_SOURCE_COLUMN: train_source,
                    "split": "train",
                }
            ),
            pd.DataFrame(
                {
                    "PATNO": test_patno,
                    BASELINE_SOURCE_COLUMN: test_source,
                    "split": "test",
                }
            ),
        ],
        ignore_index=True,
    ).sort_values("PATNO")
    baseline_source_audit.to_csv("step_08_baseline_source.csv", index=False)
    print(
        "  [Baseline source] "
        + ", ".join(
            f"{event}={count}"
            for event, count in baseline_source_audit[BASELINE_SOURCE_COLUMN]
            .value_counts()
            .items()
        )
    )

    # Drop features with > 20% missing values (computed on the training set only)
    X_train_raw = X_train_raw.dropna(thresh=len(X_train_raw) * 0.8, axis=1)
    dropped = [f for f in valid_features if f not in X_train_raw.columns]
    print(f"  [Missingness filter] dropped {len(dropped)} features: {dropped}")

    X_test_clean = X_test_raw[X_train_raw.columns]
    X_train_clean = X_train_raw.copy()

    # Save the split data with missing values, so that nested CV can refit all preprocessing and selection.
    train_unprocessed = X_train_clean.copy()
    train_unprocessed.insert(0, "PATNO", train_patno)
    train_unprocessed["Target"] = y_train
    train_unprocessed.to_csv("PPMI_4_train_unprocessed.csv", index=False)

    test_unprocessed = X_test_clean.copy()
    test_unprocessed.insert(0, "PATNO", test_patno)
    test_unprocessed["Target"] = y_test
    test_unprocessed.to_csv("PPMI_4_test_unprocessed.csv", index=False)
    print("Unprocessed split data saved for nested CV and reproducibility checks")

    # All transformations are fitted on the training set only. Race is nominal with White as
    # the reference, giving indicators for Black, Asian and Other.
    processed = fit_baseline_preprocessor(
        X_train_clean.reset_index(drop=True),
        X_test_clean.reset_index(drop=True),
    )
    X_train_filled = processed.train_filled
    X_test_filled = processed.validation_filled
    X_train_scaled = processed.train_scaled
    X_test_scaled = processed.validation_scaled
    race_encoding_audit(processed).to_csv("step_08_race_encoding.csv", index=False)
    print(
        f"  [Variable types] {len(processed.continuous_columns)} continuous, "
        f"{len(processed.discrete_columns)} discrete; race uses 3 reference-coded indicators"
    )
    # ---------------------------
    # Save the training and test sets on the original scale (not standardised)
    # ---------------------------
    train_orig = X_train_filled.copy()
    train_orig.insert(0, "PATNO", train_patno)
    train_orig["Target"] = y_train
    train_orig.to_csv("PPMI_4_train.csv", index=False)

    test_orig = X_test_filled.copy()
    test_orig.insert(0, "PATNO", test_patno)
    test_orig["Target"] = y_test
    test_orig.to_csv("PPMI_4_test.csv", index=False)

    print(
        "Original-scale training and test sets saved as PPMI_4_train.csv / PPMI_4_test.csv"
    )
    # Save the candidate feature list
    pd.DataFrame(
        {
            "feature_name": X_train_scaled.columns.tolist(),
            "index": range(1, len(X_train_scaled.columns) + 1),
        }
    ).to_csv("step_08_candidate_features_cleaned.csv", index=False)
    print(
        f"3. Preprocessing finished; {len(X_train_scaled.columns)} candidate features saved"
    )

    # ==========================================
    # 5. LASSO cross-validation
    # ==========================================
    print("\n>>> Running LASSO cross-validation (AUC-PR)...")
    # lambda = 10^[-1:0.1:3] and C = 1 / lambda, so C = 10^[-3:0.1:1]
    Cs = np.logspace(-3, 1, 41)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros((cv.get_n_splits(), len(Cs)), dtype=float)
    cv_source = X_train_clean.reset_index(drop=True)
    for fold, (fit_index, validation_index) in enumerate(cv.split(cv_source, y_train)):
        fold_data = fit_baseline_preprocessor(
            cv_source.iloc[fit_index], cv_source.iloc[validation_index]
        )
        for c_index, c_value in enumerate(Cs):
            model = LogisticRegression(
                penalty="l1",
                C=float(c_value),
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            )
            model.fit(fold_data.train_scaled, y_train[fit_index])
            scores[fold, c_index] = average_precision_score(
                y_train[validation_index],
                model.predict_proba(fold_data.validation_scaled)[:, 1],
            )
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0, ddof=1)
    n_folds = len(scores)
    se_scores = std_scores / np.sqrt(n_folds)

    # Rule: choose the C with the highest AUC-PR
    best_idx = np.argmax(mean_scores)
    best_c = Cs[best_idx]
    best_score = mean_scores[best_idx]

    print(f"  [MAX rule] best AUC-PR: {best_score:.4f}, C: {best_c:.6f}")

    # ==========================================
    # 6. Save the CV results as CSV
    # ==========================================

    # 6.1 CV score curve
    pd.DataFrame(
        {
            "C": Cs,
            "lambda": 1 / Cs,
            "log_lambda": np.log10(1 / Cs),
            "mean_score": mean_scores,
            "std_score": std_scores,
            "se_score": se_scores,
        }
    ).to_csv("step_08_cv_scores.csv", index=False)
    print("  [Saved] step_08_cv_scores.csv")

    # 6.2 Best parameters
    pd.DataFrame(
        {
            "rule": ["max"],
            "best_C": [best_c],
            "best_score": [best_score],
            "best_idx": [best_idx],
        }
    ).to_csv("step_08_cv_best_params.csv", index=False)
    print("  [Saved] step_08_cv_best_params.csv")

    # 6.3 Coefficient paths (coefficients for every C value)
    print("  [Compute] coefficient paths (all C values)...")
    coef_rows = []
    for c in Cs:
        model = LogisticRegression(
            penalty="l1", C=c, solver="liblinear", random_state=42, max_iter=5000
        )
        model.fit(X_train_scaled, y_train)
        row = {"C": c, "lambda": 1 / c, "log_lambda": np.log10(1 / c)}
        row.update(dict(zip(X_train_scaled.columns, model.coef_[0])))
        coef_rows.append(row)
    pd.DataFrame(coef_rows).to_csv("step_08_coef_path.csv", index=False)
    print("  [Saved] step_08_coef_path.csv")

    # ==========================================
    # 7. Extract and save the features selected by the MAX rule
    # ==========================================
    final_lasso = LogisticRegression(
        penalty="l1", C=best_c, solver="liblinear", random_state=42, max_iter=5000
    )
    final_lasso.fit(X_train_scaled, y_train)

    selected = pd.DataFrame(
        {"feature": X_train_scaled.columns, "coefficient": final_lasso.coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)
    selected = selected[selected["coefficient"] != 0]
    selected.to_csv("step_08_selected_features.csv", index=False)
    print(
        f"\n  [MAX rule] {len(selected)} screened candidates saved to step_08_selected_features.csv"
    )

    print(
        "\n>>> LASSO screening finished; results saved as CSV. Next, run step_09_backward_stepwise.py for backward elimination."
    )


if __name__ == "__main__":
    main()
