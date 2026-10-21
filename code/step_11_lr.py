"""Step 11: penalised logistic regression classifier.

Tuned by Optuna on 10-fold cross-validated AUC-PR, calibrated with Platt scaling
on out-of-fold predictions, and evaluated on the held-out test set.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)
import optuna
import shap
from utils import *


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class LRModel:
    """
    Logistic regression tuned by Optuna Bayesian optimisation, with OOF sigmoid calibration.

    Workflow:
      1. optimize_hyperparameters        -- Bayesian search for the best hyperparameters (objective: AUC-PR)
      2. train_final_model              -- fit the final model on the full training set and a sigmoid calibrator on 10-fold OOF predictions
      3. get_cross_validation_performance -- 10-fold CV metrics on the training set at fixed global thresholds (thresholds and calibrator come from the same OOF predictions, so the estimates are optimistic and descriptive only)
      4. evaluate_feature_importance      -- coefficients, SHAP and permutation importance, with plots
      5. evaluate_and_save_results        -- calibrated performance on the test set
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None  # final fitted model
        self.calibrator = None  # OOF sigmoid (Platt) probability calibrator
        self.best_params = None  # best hyperparameters from the Optuna search
        self.default = 0.5  # default classification threshold
        self.f1_median = None  # global F1-optimal threshold
        self.youden_median = None  # global Youden-optimal threshold
        self.trial_count = 0  # current trial number shown by the progress callback

    # ----------------------------------------------------------
    # Model construction
    # ----------------------------------------------------------

    def create_model(self, params):
        """
        Create a LogisticRegression instance from a hyperparameter list.

        params : [penalty, solver, C, l1_ratio]
            penalty  : regularisation type ('l1', 'l2', 'elasticnet' or None)
            solver   : solver (must be compatible with the penalty)
            C        : inverse regularisation strength (smaller means stronger)
            l1_ratio : elastic-net mixing ratio (used only when penalty='elasticnet')
        """
        penalty, solver, C, l1_ratio = params

        model_params = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": 10000,
            "class_weight": "balanced",  # handle class imbalance
            "random_state": self.random_state,
        }

        # Elastic net also needs l1_ratio
        if penalty == "elasticnet":
            model_params["l1_ratio"] = l1_ratio

        return LogisticRegression(**model_params)

    # ----------------------------------------------------------
    # Optuna objective
    # ----------------------------------------------------------

    def objective_function(self, trial, X, y, folds, study):
        """
        Objective function for one Optuna trial.

        Valid penalty/solver pairs are encoded as one categorical parameter, which
        avoids sampling problems with a conditional search space.
        Objective: mean AUC-PR over 10-fold cross-validation (higher is better).
        """
        self.trial_count += 1

        # Encode the valid penalty/solver pairs as one categorical parameter
        param_combination = trial.suggest_categorical(
            "param_combination",
            [
                "l1_liblinear",
                "l1_saga",
                "l2_lbfgs",
                "l2_newton-cg",
                "l2_sag",
                "l2_saga",
                "l2_liblinear",
                "elasticnet_saga",
                "none_lbfgs",
                "none_newton-cg",
                "none_sag",
                "none_saga",
            ],
        )

        penalty, solver = param_combination.split("_")
        if penalty == "none":
            penalty = None  # sklearn expects None, not the string 'none'

        C = trial.suggest_float("C", 0.0001, 1, log=True)

        # Only elastic net needs l1_ratio
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:
            l1_ratio = 0.5  # placeholder, not used

        model = self.create_model([penalty, solver, C, l1_ratio])

        # 10-fold cross-validation: mean of the per-fold AUC-PR
        auc_scores = []
        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model.fit(X_train_fold, y_train_fold)
            y_val_prob = model.predict_proba(X_val_fold)[:, 1]
            auc_scores.append(average_precision_score(y_val_fold, y_val_prob))

        return np.mean(auc_scores)

    # ----------------------------------------------------------
    # Hyperparameter optimisation
    # ----------------------------------------------------------

    def optimize_hyperparameters(self, X, y, n_calls=100):
        """
        Search for the best hyperparameters with Optuna (TPE sampler).

        Parameters
        ----
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        n_calls : int, default 100
            Number of Optuna trials.
        """
        print("Starting LR Bayesian hyperparameter optimisation...")
        print("=" * 80)

        # Indices of the 10-fold stratified cross-validation
        folds = stratified_cross_validation(
            X, y, n_folds=10, random_state=self.random_state
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        self.trial_count = 0

        def callback(study, trial):
            """Print progress after each trial."""
            if trial.state != optuna.trial.TrialState.COMPLETE or not trial.params:
                return

            params_list = []
            if "param_combination" in trial.params:
                penalty_str, solver = trial.params["param_combination"].split("_")
                display_penalty = "None" if penalty_str == "none" else penalty_str
                params_list.extend([f"penalty={display_penalty}", f"solver={solver}"])
            if "C" in trial.params:
                params_list.append(f"C={trial.params['C']:.6f}")
            # Show l1_ratio only for elastic net
            if "l1_ratio" in trial.params and penalty_str == "elasticnet":
                params_list.append(f"l1_ratio={trial.params['l1_ratio']:.4f}")

            print(f"Trial: {self.trial_count}/{n_calls}")
            print(f"Parameters: {', '.join(params_list)}")
            print(f"AUC-PR: {trial.value:.6f}")
            print(
                f"Best AUC-PR (trial {study.best_trial.number + 1}): {study.best_value:.6f}"
            )
            print()

        study.optimize(
            lambda trial: self.objective_function(trial, X, y, folds, study),
            n_trials=n_calls,
            callbacks=[callback],
        )

        print("=" * 80)
        print("Hyperparameter search finished")

        # Decode the best parameters
        best_trial = study.best_trial
        param_combination = best_trial.params["param_combination"]
        penalty, solver = param_combination.split("_")
        if penalty == "none":
            penalty = None

        self.best_params = {
            "penalty": penalty,
            "solver": solver,
            "C": best_trial.params["C"],
            "l1_ratio": best_trial.params.get("l1_ratio", 0.5),
        }

        # Print only the parameters that take effect
        display_params = {
            "penalty": "None" if penalty is None else penalty,
            "solver": solver,
            "C": self.best_params["C"],
        }
        if penalty == "elasticnet":
            display_params["l1_ratio"] = self.best_params["l1_ratio"]

        print(f"Best parameters: {display_params}")
        print(f"Best AUC-PR: {best_trial.value:.6f}")

        best_params_df = pd.DataFrame(
            [
                {
                    "model": "LR",
                    "best_trial": best_trial.number + 1,
                    "best_auc_pr": best_trial.value,
                    "penalty": "None" if penalty is None else penalty,
                    "solver": solver,
                    "C": self.best_params["C"],
                    "l1_ratio": (
                        self.best_params["l1_ratio"] if penalty == "elasticnet" else ""
                    ),
                }
            ]
        )
        best_params_df.to_csv(
            "step_11_lr_best_hyperparameters.csv", index=False, encoding="utf-8-sig"
        )
        print("LR best hyperparameters saved to: step_11_lr_best_hyperparameters.csv")

        return study

    # ----------------------------------------------------------
    # Final model and optimal thresholds
    # ----------------------------------------------------------

    def train_final_model(self, X, y, feature_cols=None):
        """
        Fit the final model on the full training set with the best hyperparameters.
        Also fit a sigmoid probability calibrator on 10-fold OOF predictions and estimate the global F1- and Youden-optimal thresholds.
        """
        print("Fitting the final LR model and the OOF sigmoid calibrator...")

        folds = stratified_cross_validation(
            X, y, n_folds=10, random_state=self.random_state
        )
        oof_raw_probs = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_model = self.create_model(
                [
                    self.best_params["penalty"],
                    self.best_params["solver"],
                    self.best_params["C"],
                    self.best_params["l1_ratio"],
                ]
            )
            fold_model.fit(X_train_fold, y_train_fold)
            oof_raw_probs[val_idx] = fold_model.predict_proba(X_val_fold)[:, 1]

        # Fit the Platt (sigmoid) probability calibrator
        # logit(P_cal) = a * logit(P_raw) + b
        oof_logits = logit(oof_raw_probs).reshape(-1, 1)
        self.calibrator = LogisticRegression(
            penalty=None, solver="lbfgs", random_state=self.random_state
        )
        self.calibrator.fit(oof_logits, y)

        cal_a = float(self.calibrator.coef_[0][0])
        cal_b = float(self.calibrator.intercept_[0])
        print(f"Sigmoid calibrator fitted: a={cal_a:.4f}, b={cal_b:.4f}")

        pd.DataFrame(
            [
                {
                    "calibration_method": "platt_10fold_oof",
                    "n_folds": 10,
                    "cal_a": cal_a,
                    "cal_b": cal_b,
                }
            ]
        ).to_csv("step_11_lr_calibration_parameters.csv", index=False)
        print(
            "LR calibration parameters saved to: step_11_lr_calibration_parameters.csv"
        )

        # Calibrated OOF probabilities
        oof_cal_probs = self.calibrator.predict_proba(oof_logits)[:, 1]

        # Global optimal thresholds from all OOF probabilities (avoids small-sample noise of per-fold thresholds)
        self.default = 0.5
        self.f1_median = float(f1_score_threshold(y, oof_cal_probs))
        self.youden_median = float(youden_threshold(y, oof_cal_probs))

        print("\n" + "=" * 60)
        print("Calibrated global classification thresholds (from OOF predictions):")
        print("=" * 60)
        print(f"1. Default threshold:        0.5000")
        print(f"2. F1-optimal threshold:     {self.f1_median:.4f}")
        print(f"3. Youden-optimal threshold: {self.youden_median:.4f}")
        print("=" * 60)

        # Fit the final base model on the full training set
        self.model = self.create_model(
            [
                self.best_params["penalty"],
                self.best_params["solver"],
                self.best_params["C"],
                self.best_params["l1_ratio"],
            ]
        )
        self.model.fit(X, y)

        # Print the intercept and save the calibration parameters (needed for the nomogram)
        raw_intercept = float(self.model.intercept_[0])
        print(f"Logistic raw intercept: {raw_intercept:.6f}")

        # Save the raw intercept and calibration parameters to CSV
        cal_df = pd.DataFrame(
            [{"raw_intercept": raw_intercept, "cal_a": cal_a, "cal_b": cal_b}]
        )
        cal_df.to_csv("step_11_lr_nomogram_intercept.csv", index=False)
        print(
            "Logistic intercept and calibration parameters saved to step_11_lr_nomogram_intercept.csv"
        )

        if feature_cols is not None:
            pd.DataFrame(
                {"feature": feature_cols, "coefficient": self.model.coef_[0]}
            ).to_csv("step_11_lr_model_coefficients.csv", index=False)
            print(
                "Logistic standardised coefficients saved to step_11_lr_model_coefficients.csv"
            )

    # ----------------------------------------------------------
    # Cross-validated performance on the training set
    # ----------------------------------------------------------

    def get_cross_validation_performance(self, X, y):
        """
        Run 10-fold cross-validation on the training set and report the mean and SD of each
        metric at fixed global thresholds (default, global F1, global Youden); thresholds are not re-selected within validation folds.
        Note: the calibrator and thresholds were fitted on the same 10-fold OOF predictions, so threshold-dependent metrics are optimistic and descriptive only.
        """
        print(
            "Computing 10-fold CV performance on the training set (calibrated, fixed global thresholds)..."
        )

        folds = stratified_cross_validation(
            X, y, n_folds=10, random_state=self.random_state
        )
        fold_results = {"default": [], "f1": [], "youden": []}

        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_model = self.create_model(
                [
                    self.best_params["penalty"],
                    self.best_params["solver"],
                    self.best_params["C"],
                    self.best_params["l1_ratio"],
                ]
            )
            fold_model.fit(X_train_fold, y_train_fold)
            y_val_raw_prob = fold_model.predict_proba(X_val_fold)[:, 1]
            # Apply the calibrator
            y_val_prob = self.calibrator.predict_proba(
                logit(y_val_raw_prob).reshape(-1, 1)
            )[:, 1]

            # Metrics at the three fixed global thresholds (not re-selected within folds)
            for thresh_type, threshold in [
                ("default", self.default),
                ("f1", self.f1_median),
                ("youden", self.youden_median),
            ]:
                y_pred = (y_val_prob >= threshold).astype(int)
                metrics = calculate_all_metrics(
                    y_val_fold, y_pred, y_val_prob, threshold
                )
                fold_results[thresh_type].append(metrics)

        # Summarise the fold metrics (mean and SD)
        summary_results = {}
        for thresh_type in ["default", "f1", "youden"]:
            metrics_df = pd.DataFrame(fold_results[thresh_type])
            summary_results[f"train_{thresh_type}_mean"] = metrics_df.mean().to_dict()
            summary_results[f"train_{thresh_type}_std"] = metrics_df.std().to_dict()

        return summary_results

    # ----------------------------------------------------------
    # Feature importance
    # ----------------------------------------------------------

    def evaluate_feature_importance(self, X, y, feature_names):
        """
        Compute and plot three feature-importance measures:
          1. Model coefficients (absolute values)
          2. SHAP (LinearExplainer)
          3. Permutation importance

        Parameters
        ----
        X : np.ndarray
        y : np.ndarray
        feature_names : list of str
            Feature names (abbreviations) matching the columns of X.
        """
        print("Computing LR feature importance...")

        importances = {}

        # 1. Logistic regression coefficients (absolute values)
        if hasattr(self.model, "coef_"):
            importances["Coefficients"] = np.abs(self.model.coef_[0])

        # 2. SHAP (LinearExplainer on a training subset for speed)
        sample_size = min(500, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]

        explainer = shap.LinearExplainer(self.model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        importances["SHAP"] = np.mean(np.abs(shap_values), axis=0)
        plot_shap_summary(shap_values, X_sample, feature_names, "LR")

        # 3. Permutation importance
        perm_importance = get_permutation_importance(
            self.model, X, y, feature_names, self.random_state
        )
        importances["Permutation"] = perm_importance

        plot_feature_importance_comparison(importances, feature_names, "LR")

        return importances

    # ----------------------------------------------------------
    # Test-set performance
    # ----------------------------------------------------------

    def evaluate_and_save_results(self, X_test, y_test):
        """
        Compute metrics on the independent test set at the default, F1-optimal and Youden-optimal thresholds (with sigmoid calibration).

        Returns
        ----
        test_performance : dict  keys 'test_default', 'test_f1', 'test_youden'
        y_prob_raw : np.ndarray  uncalibrated test-set probabilities
        y_prob : np.ndarray      calibrated test-set probabilities
        """
        print("Computing test-set performance (with sigmoid calibration)...")
        y_prob_raw = self.model.predict_proba(X_test)[:, 1]
        y_prob = self.calibrator.predict_proba(logit(y_prob_raw).reshape(-1, 1))[:, 1]
        print(
            f"Mean test-set probability, raw: {y_prob_raw.mean():.4f}, calibrated: {y_prob.mean():.4f}"
        )

        test_performance = {}

        for thresh_type, threshold in [
            ("test_default", self.default),
            ("test_f1", self.f1_median),
            ("test_youden", self.youden_median),
        ]:
            y_pred = (y_prob >= threshold).astype(int)
            test_performance[thresh_type] = calculate_all_metrics(
                y_test, y_pred, y_prob, threshold
            )

        return test_performance, y_prob_raw, y_prob


# ============================================================
# Main workflow
# ============================================================


def main():
    np.random.seed(42)

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    train_data, test_data, feature_name_mapping, scaler = load_data(
        "PPMI_4_train.csv",
        "PPMI_4_test.csv",
        "PPMI_feature_mapping.csv",
        "step_09_selected_features.csv",
        return_scaler=True,
    )

    # ----------------------------------------------------------
    # 2. Separate features and labels; convert to numpy arrays
    # ----------------------------------------------------------
    X_train, y_train, _ = prepare_data(train_data)
    X_test, y_test, test_patno = prepare_data(test_data)

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Replace column names by their abbreviations (unmapped names are kept)
    raw_feature_cols = [
        col for col in train_data.columns if col not in ["PATNO", "Target"]
    ]
    pd.DataFrame(
        {"feature": raw_feature_cols, "mean": scaler.mean_, "scale": scaler.scale_}
    ).to_csv("step_11_lr_scaler_parameters.csv", index=False)
    print(
        "LR training-set scaler parameters saved to: step_11_lr_scaler_parameters.csv"
    )

    feature_names = [feature_name_mapping.get(col, col) for col in raw_feature_cols]

    # ----------------------------------------------------------
    # 3. Bayesian hyperparameter optimisation
    # ----------------------------------------------------------
    lr_model = LRModel(random_state=42)
    lr_model.optimize_hyperparameters(X_train, y_train)

    # ----------------------------------------------------------
    # 4. Final model and optimal classification thresholds
    # ----------------------------------------------------------
    lr_model.train_final_model(X_train, y_train, feature_cols=raw_feature_cols)

    # ----------------------------------------------------------
    # 5. 10-fold CV performance on the training set
    # ----------------------------------------------------------
    train_performance = lr_model.get_cross_validation_performance(X_train, y_train)

    # ----------------------------------------------------------
    # 6. Feature importance
    # ----------------------------------------------------------
    lr_model.evaluate_feature_importance(X_train, y_train, feature_names)

    # ----------------------------------------------------------
    # 7. Test-set evaluation
    # ----------------------------------------------------------
    test_performance, y_prob_raw, y_prob = lr_model.evaluate_and_save_results(
        X_test, y_test
    )

    # Plot ROC and PR curves
    plot_roc_pr_curves(
        y_test,
        y_prob,
        "LR",
        calibration_method="platt_10fold_oof",
        patient_ids=test_patno.to_numpy(),
        raw_prob=y_prob_raw,
        model_version="step_11_lr_oof_platt_v1",
    )

    # ----------------------------------------------------------
    # 8. Save and print results
    # ----------------------------------------------------------
    train_df = pd.DataFrame(train_performance)
    train_df.to_csv("step_11_lr_train_performance.csv")
    print(f"\nTraining-set performance saved to: step_11_lr_train_performance.csv")

    test_df = pd.DataFrame(test_performance)
    test_df.to_csv("step_11_lr_test_performance.csv")
    print(f"Test-set performance saved to: step_11_lr_test_performance.csv")

    print("\n" + "=" * 80)
    print("LR training-set performance (mean ± SD):")
    print("=" * 80)
    print(train_df.round(4))

    print("\n" + "=" * 80)
    print("LR test-set performance:")
    print("=" * 80)
    print(test_df.round(4))


if __name__ == "__main__":
    main()
