"""Step 13: back-transform the calibrated logistic regression to original units for the nomogram.

Combines the step 11 coefficients, training-set scaling and Platt calibration
into one original-unit formula, checks it against the saved probabilities and
writes step_13_lr_nomogram_parameters.csv for the R nomogram (step 14).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# Save the model coefficients (read by the R nomogram script)
# ==============================================================================
def save_model_coefficients(
    coefficients,
    intercept,
    feature_names,
    feature_labels,
    feature_mins,
    feature_maxs,
    save_path="step_13_lr_nomogram_parameters.csv",
):
    """
    Save the LR coefficients, intercept and feature ranges to CSV.

    Columns
    ----
    Feature     : original column name (as in the training data)
    Label       : display name (nomogram axis label)
    Coefficient : regression coefficient on the original scale
    Min_Value   : minimum of the feature in the training set
    Max_Value   : maximum of the feature in the training set
    The intercept is a separate row (Feature='Intercept') with empty Min/Max.
    """
    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Label": feature_labels,
            "Coefficient": coefficients,
            "Min_Value": feature_mins,
            "Max_Value": feature_maxs,
        }
    )

    intercept_row = pd.DataFrame(
        {
            "Feature": ["Intercept"],
            "Label": ["Intercept"],
            "Coefficient": [intercept],
            "Min_Value": [np.nan],
            "Max_Value": [np.nan],
        }
    )

    coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
    coef_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\nModel parameters saved to: {save_path}")
    print("=" * 70)
    print(f"{'Feature':<20}{'Label':<22}{'Coefficient':>14}")
    print("-" * 70)
    print(f"{'Intercept':<20}{'Intercept':<22}{intercept:>14.6f}")
    for fname, flabel, coef in zip(feature_names, feature_labels, coefficients):
        print(f"{fname:<20}{flabel:<22}{coef:>14.6f}")
    print("=" * 70)

    return coef_df


def main():
    np.random.seed(42)

    # 1. Read the original-scale training set
    train_data = pd.read_csv("PPMI_4_train.csv")

    # 2. Read the feature-name mapping (original name -> display name)
    mapping_df = pd.read_csv("PPMI_feature_mapping.csv")
    feature_name_mapping = dict(zip(mapping_df["Feature"], mapping_df["Abbreviation"]))

    # 3. Read the final predictors of the two-stage selection
    selected_df = pd.read_csv("step_09_selected_features.csv")
    selected_features = selected_df["feature"].tolist()
    print(f"Read {len(selected_features)} features from step_09_selected_features.csv:")
    print(selected_features)

    # Unmapped features keep their original names
    feature_labels = [feature_name_mapping.get(f, f) for f in selected_features]

    target_column = "Target" if "Target" in train_data.columns else "MCI"

    X_train = train_data[selected_features].values
    y_train = train_data[target_column].values

    # 4. Read the tuned penalised LR coefficients, calibration parameters and scaler from step 11
    cal_info = pd.read_csv("step_11_lr_nomogram_intercept.csv").iloc[0]
    raw_intercept = float(cal_info["raw_intercept"])
    cal_a = float(cal_info["cal_a"])
    cal_b = float(cal_info["cal_b"])

    coef_info = pd.read_csv("step_11_lr_model_coefficients.csv")
    coef_map = dict(zip(coef_info["feature"], coef_info["coefficient"]))

    scaler_info = pd.read_csv("step_11_lr_scaler_parameters.csv")
    scaler_info = scaler_info.set_index("feature")

    missing_scaler_features = set(selected_features) - set(scaler_info.index)
    if missing_scaler_features:
        raise ValueError(
            f"Scaler parameters lack features: {sorted(missing_scaler_features)}"
        )

    # Use the mean and population SD (ddof=0) of the StandardScaler fitted in training
    std_coefs = np.array([coef_map[f] for f in selected_features])
    means = scaler_info.loc[selected_features, "mean"].to_numpy(dtype=float)
    stds = scaler_info.loc[selected_features, "scale"].to_numpy(dtype=float)
    feature_mins = [train_data[f].min() for f in selected_features]
    feature_maxs = [train_data[f].max() for f in selected_features]

    # 5. Back-transform to original-unit coefficients and fold in the sigmoid calibration
    # beta_raw = cal_a * (beta_std / std)
    # intercept_raw = cal_a * raw_intercept + cal_b - sum(cal_a * beta_std * mean / std)
    raw_coefficients = (cal_a * std_coefs) / stds
    raw_intercept_final = (cal_a * raw_intercept + cal_b) - np.sum(
        (cal_a * std_coefs * means) / stds
    )

    # Check per participant that the original-unit formula reproduces the calibrated standardised LR
    standardized_logits = raw_intercept + ((X_train - means) / stds) @ std_coefs
    calibrated_probs = 1 / (1 + np.exp(-(cal_a * standardized_logits + cal_b)))
    nomogram_logits = raw_intercept_final + X_train @ raw_coefficients
    nomogram_probs = 1 / (1 + np.exp(-nomogram_logits))

    max_abs_diff = float(np.max(np.abs(calibrated_probs - nomogram_probs)))
    tolerance = 1e-10
    validation_passed = max_abs_diff < tolerance

    pd.DataFrame(
        [
            {
                "n_samples": len(X_train),
                "tolerance": tolerance,
                "max_absolute_probability_difference": max_abs_diff,
                "validation_passed": validation_passed,
            }
        ]
    ).to_csv("step_13_lr_nomogram_validation.csv", index=False)

    if not validation_passed:
        raise ValueError(
            f"Nomogram probability check failed: max_abs_diff={max_abs_diff:.3e}"
        )

    print(
        f"\nStep 11 penalised and calibration parameters back-transformed to original units:"
    )
    print(f"Calibration: a={cal_a:.4f}, b={cal_b:.4f}")
    print(f"Final nomogram intercept: {raw_intercept_final:.6f}")
    print(f"Per-participant probability check passed: max_abs_diff={max_abs_diff:.3e}")

    # 6. Save the coefficients (with intercept and original ranges)
    save_model_coefficients(
        coefficients=raw_coefficients,
        intercept=raw_intercept_final,
        feature_names=selected_features,
        feature_labels=feature_labels,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        save_path="step_13_lr_nomogram_parameters.csv",
    )


if __name__ == "__main__":
    main()
