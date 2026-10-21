#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 09: backward elimination of the LASSO-screened candidates (second selection stage).

Tasks:
1. Read the candidates kept by the step 08 LASSO screen (MAX rule);
2. Read the unstandardised training and test sets (PPMI_4_train.csv / PPMI_4_test.csv);
3. Fit a multivariable logistic regression on the training set only and apply backward elimination;
4. Keep variables with two-sided Wald P < 0.05: each iteration removes the variable with the largest P >= 0.05 until all remaining variables are significant;
5. Write the final predictors, the elimination log with full statistics and the effect-size table;
6. Write the training-set mean, SD and 5th-95th percentiles of the final predictors for the nomogram back-transformation.

Outputs:
- step_09_selected_features.csv: final predictors
- step_09_stepwise_history.csv: backward-elimination log
- step_09_multivariable_pvalues.csv: final multivariable logistic regression (OR, 95% CI, P value)
- step_09_nomogram_feature_stats.csv: training-set statistics of the final predictors on the original scale
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

CODE_DIR = Path(__file__).resolve().parent

FINAL_FEATURE_ORDER = [
    "scopa",
    "NP1COG",
    "rem",
    "DVT_SFTANIM",
    "MIA_STRIATUM_mean",
    "SEX",
    "LEDD",
    "upsit_pctl",
]


def main():
    """Run the backward elimination and save the final predictors and statistics."""
    print("=" * 70)
    print(">>> Step 09: backward elimination of the screened candidates")
    print("=" * 70)

    # =========================================================================
    # 1. Load data and candidate features
    # =========================================================================
    train_raw_path = CODE_DIR / "PPMI_4_train.csv"
    test_raw_path = CODE_DIR / "PPMI_4_test.csv"
    lasso_selected_path = CODE_DIR / "step_08_selected_features.csv"

    print(f"1. Data and candidate features:")
    print(f"   - training set (original scale): {train_raw_path.name}")
    print(f"   - test set (original scale): {test_raw_path.name}")
    print(f"   - LASSO-screened features: {lasso_selected_path.name}")

    train_raw = pd.read_csv(train_raw_path)
    test_raw = pd.read_csv(test_raw_path)
    lasso_selected = pd.read_csv(lasso_selected_path)["feature"].tolist()

    print(f"   - candidates from the LASSO MAX screen (N={len(lasso_selected)}):")
    print(f"     {lasso_selected}")

    y_train = train_raw["Target"].values
    y_test = test_raw["Target"].values
    X_train_raw = train_raw[lasso_selected].copy()

    # =========================================================================
    # 2. Multivariable backward elimination (keep P < 0.05)
    # =========================================================================
    print("\n2. Backward elimination (keep variables with Wald P < 0.05):")
    print("-" * 70)
    print(
        f"{'Step':<6}{'Action / Dropped Feature':<26}{'Max P-value':<14}{'AIC':<12}{'Remaining':<10}"
    )
    print("-" * 70)

    current_features = list(lasso_selected)
    history = []
    step_num = 0

    while len(current_features) > 1:
        step_num += 1
        X_const = sm.add_constant(X_train_raw[current_features])
        model = sm.Logit(y_train, X_const).fit(disp=False)

        # P values of the features (excluding the intercept)
        pvals = model.pvalues.iloc[1:]
        max_p = pvals.max()
        max_feat = pvals.idxmax()
        aic = model.aic
        bic = model.bic

        if max_p >= 0.05:
            history.append(
                {
                    "step": step_num,
                    "action": f"Dropped {max_feat}",
                    "dropped_feature": max_feat,
                    "max_p_value": round(float(max_p), 6),
                    "aic": round(float(aic), 4),
                    "bic": round(float(bic), 4),
                    "n_remaining": len(current_features) - 1,
                }
            )
            print(
                f"{step_num:<6}{max_feat:<26}{max_p:<14.4f}{aic:<12.2f}{len(current_features) - 1:<10}"
            )
            current_features.remove(max_feat)
        else:
            history.append(
                {
                    "step": step_num,
                    "action": "Converged (All P < 0.05)",
                    "dropped_feature": "None",
                    "max_p_value": round(float(max_p), 6),
                    "aic": round(float(aic), 4),
                    "bic": round(float(bic), 4),
                    "n_remaining": len(current_features),
                }
            )
            print(
                f"{step_num:<6}{'None (Converged)':<26}{max_p:<14.4f}{aic:<12.2f}{len(current_features):<10}"
            )
            break

    # Use a fixed clinical display order so that the LASSO coefficient order cannot change the random feature sampling of tree models
    selected_feature_set = set(current_features)
    current_features = [f for f in FINAL_FEATURE_ORDER if f in selected_feature_set]
    current_features.extend(sorted(selected_feature_set - set(FINAL_FEATURE_ORDER)))

    print("-" * 70)
    print(
        f"\n3. Elimination converged: {len(current_features)} independently significant predictors:"
    )
    print(f"   {current_features}")

    # =========================================================================
    # 3. Fit the final multivariable logistic model and compute effect sizes (OR, 95% CI, P)
    # =========================================================================
    final_const = sm.add_constant(X_train_raw[current_features])
    final_model = sm.Logit(y_train, final_const).fit(disp=False)

    ci = final_model.conf_int()
    summary_rows = []
    for feat in current_features:
        coef = final_model.params[feat]
        se = final_model.bse[feat]
        z = final_model.tvalues[feat]
        pval = final_model.pvalues[feat]
        or_val = np.exp(coef)
        ci_lower = np.exp(ci.loc[feat, 0])
        ci_upper = np.exp(ci.loc[feat, 1])
        summary_rows.append(
            {
                "feature": feat,
                "coef": coef,
                "std_err": se,
                "z_value": z,
                "p_value": pval,
                "odds_ratio": or_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # =========================================================================
    # 4. Summary statistics and selected-feature list
    # =========================================================================
    # Training-set statistics of the final predictors on the original scale
    stats_df = pd.DataFrame(
        {
            "feature": current_features,
            "mean": [train_raw[f].mean() for f in current_features],
            "std": [train_raw[f].std() for f in current_features],
            "range_low": [train_raw[f].quantile(0.05) for f in current_features],
            "range_high": [train_raw[f].quantile(0.95) for f in current_features],
        }
    )

    # Selected-feature list
    selected_features_df = pd.DataFrame(
        {
            "feature": current_features,
            "p_value": [final_model.pvalues[f] for f in current_features],
            "multivariable_coef": [final_model.params[f] for f in current_features],
            "odds_ratio": [np.exp(final_model.params[f]) for f in current_features],
        }
    )

    # =========================================================================
    # 5. Save the results
    # =========================================================================
    selected_features_df.to_csv(CODE_DIR / "step_09_selected_features.csv", index=False)
    pd.DataFrame(history).to_csv(CODE_DIR / "step_09_stepwise_history.csv", index=False)
    summary_df.to_csv(CODE_DIR / "step_09_multivariable_pvalues.csv", index=False)
    stats_df.to_csv(CODE_DIR / "step_09_nomogram_feature_stats.csv", index=False)

    print("\n4. Outputs saved:")
    print(
        f"   - step_09_selected_features.csv ({len(current_features)} final predictors)"
    )
    print("   - step_09_stepwise_history.csv (elimination log)")
    print("   - step_09_multivariable_pvalues.csv (effect sizes and P values)")
    print(
        "   - step_09_nomogram_feature_stats.csv (original-scale means and percentiles)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
