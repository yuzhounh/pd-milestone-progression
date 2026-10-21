"""Step 11: support vector machine classifier.

Tuned by Optuna on 10-fold cross-validated AUC-PR, calibrated with Platt scaling
on out-of-fold predictions, and evaluated on the held-out test set.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.svm import SVC
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


# Global font settings (as in utils.py, repeated so that the script also works on its own)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = True


class SVMModel:
    """
    Support vector machine classifier tuned by Optuna Bayesian optimisation, with OOF sigmoid calibration.

    Workflow:
      1. optimize_hyperparameters        -- Bayesian search for the best hyperparameters (objective: AUC-PR)
      2. train_final_model              -- fit the final model and a sigmoid calibrator on 10-fold OOF predictions
      3. get_cross_validation_performance -- 10-fold CV metrics on the training set at three thresholds
      4. evaluate_feature_importance      -- SHAP and permutation importance, with plots (plus weights for a linear kernel)
      5. evaluate_and_save_results        -- test-set performance at three thresholds
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None  # final fitted model
        self.calibrator = None  # OOF sigmoid (Platt) probability calibrator
        self.best_params = None  # best hyperparameters from the Optuna search
        self.default = 0.5  # default classification threshold
        self.f1_median = (
            None  # global F1-optimal threshold (calibrated OOF probabilities)
        )
        self.youden_median = (
            None  # global Youden-optimal threshold (calibrated OOF probabilities)
        )
        self.trial_count = 0  # current trial number shown by the progress callback

    # ----------------------------------------------------------
    # Model construction
    # ----------------------------------------------------------

    def create_model(self, params):
        """
        Create an SVC instance from a hyperparameter list.

        params : [kernel, C, gamma, degree, coef0]
            kernel : kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C      : penalty parameter (larger tolerates fewer misclassifications)
            gamma  : kernel coefficient (rbf / poly / sigmoid)
            degree : degree of the polynomial kernel (poly only)
            coef0  : independent term of the kernel (poly / sigmoid)
        """
        kernel, C, gamma, degree, coef0 = params

        model_params = {
            "C": C,
            "kernel": kernel,
            "class_weight": "balanced",  # handle class imbalance
            "random_state": self.random_state,
            "probability": True,  # enable predict_proba
            "max_iter": 10000,
        }

        # Add the parameters used by the chosen kernel
        if kernel in ["rbf", "poly", "sigmoid"]:
            model_params["gamma"] = gamma
        if kernel == "poly":
            model_params["degree"] = degree
            model_params["coef0"] = coef0
        elif kernel == "sigmoid":
            model_params["coef0"] = coef0

        return SVC(**model_params)

    # ----------------------------------------------------------
    # Optuna objective
    # ----------------------------------------------------------

    def objective_function(self, trial, X, y, folds, study):
        """
        Objective function for one Optuna trial.

        Hyperparameters are sampled conditionally on the kernel to avoid unused combinations.
        Objective: mean AUC-PR over 10-fold cross-validation (higher is better).
        """
        self.trial_count += 1

        kernel = trial.suggest_categorical(
            "kernel", ["linear", "rbf", "poly", "sigmoid"]
        )
        C = trial.suggest_float("C", 0.0001, 1, log=True)

        # Sample conditionally on the kernel type
        if kernel == "linear":
            gamma, degree, coef0 = "scale", 3, 0.0
        elif kernel == "rbf":
            gamma = trial.suggest_float("gamma", 0.0001, 1, log=True)
            degree, coef0 = 3, 0.0
        elif kernel == "poly":
            gamma = trial.suggest_float("gamma", 0.0001, 1, log=True)
            degree = trial.suggest_int("degree", 2, 5)
            coef0 = trial.suggest_float("coef0", 0.0, 10.0)
        else:  # sigmoid
            gamma = trial.suggest_float("gamma", 0.0001, 1, log=True)
            coef0 = trial.suggest_float("coef0", 0.0, 10.0)
            degree = 3

        model = self.create_model([kernel, C, gamma, degree, coef0])

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
        print("Starting SVM Bayesian hyperparameter optimisation...")
        print("=" * 80)

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

            kernel = trial.params.get("kernel", "unknown")
            params_list = [f"kernel={kernel}"]

            if "C" in trial.params:
                params_list.append(f"C={trial.params['C']:.6f}")
            if kernel in ["rbf", "poly", "sigmoid"] and "gamma" in trial.params:
                params_list.append(f"gamma={trial.params['gamma']:.6f}")
            if kernel == "poly" and "degree" in trial.params:
                params_list.append(f"degree={trial.params['degree']}")
            if kernel in ["poly", "sigmoid"] and "coef0" in trial.params:
                params_list.append(f"coef0={trial.params['coef0']:.4f}")

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

        # Decode the best parameters and fill defaults for unused fields
        best_trial = study.best_trial
        kernel = best_trial.params["kernel"]
        self.best_params = {"kernel": kernel, "C": best_trial.params["C"]}

        if kernel == "linear":
            self.best_params.update({"gamma": "scale", "degree": 3, "coef0": 0.0})
        elif kernel == "rbf":
            self.best_params.update(
                {"gamma": best_trial.params["gamma"], "degree": 3, "coef0": 0.0}
            )
        elif kernel == "poly":
            self.best_params.update(
                {
                    "gamma": best_trial.params["gamma"],
                    "degree": best_trial.params["degree"],
                    "coef0": best_trial.params["coef0"],
                }
            )
        else:  # sigmoid
            self.best_params.update(
                {
                    "gamma": best_trial.params["gamma"],
                    "degree": 3,
                    "coef0": best_trial.params["coef0"],
                }
            )

        # Print only the parameters that take effect
        display_params = {"kernel": kernel, "C": self.best_params["C"]}
        if kernel in ["rbf", "poly", "sigmoid"]:
            display_params["gamma"] = self.best_params["gamma"]
        if kernel == "poly":
            display_params["degree"] = self.best_params["degree"]
            display_params["coef0"] = self.best_params["coef0"]
        elif kernel == "sigmoid":
            display_params["coef0"] = self.best_params["coef0"]

        print(f"Best parameters: {display_params}")
        print(f"Best AUC-PR: {best_trial.value:.6f}")

        best_params_df = pd.DataFrame(
            [
                {
                    "model": "SVM",
                    "best_trial": best_trial.number + 1,
                    "best_auc_pr": best_trial.value,
                    "kernel": self.best_params["kernel"],
                    "C": self.best_params["C"],
                    "gamma": self.best_params["gamma"],
                    "degree": self.best_params["degree"],
                    "coef0": self.best_params["coef0"],
                }
            ]
        )
        best_params_df.to_csv(
            "step_11_svm_best_hyperparameters.csv", index=False, encoding="utf-8-sig"
        )
        print("SVM best hyperparameters saved to: step_11_svm_best_hyperparameters.csv")

        return study

    # ----------------------------------------------------------
    # Final model and optimal thresholds
    # ----------------------------------------------------------

    def train_final_model(self, X, y):
        """
        Fit the final model on the full training set with the best hyperparameters.
        Also fit a sigmoid calibrator on 10-fold OOF predictions and estimate the global F1- and Youden-optimal thresholds on the calibrated OOF probabilities.

        Parameters
        ----
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        print("Fitting the final SVM model and the OOF sigmoid calibrator...")

        self.model = self.create_model(
            [
                self.best_params["kernel"],
                self.best_params["C"],
                self.best_params["gamma"],
                self.best_params["degree"],
                self.best_params["coef0"],
            ]
        )
        self.model.fit(X, y)

        # Global optimal thresholds from 10-fold OOF predictions (no per-fold small-sample leakage)
        folds = stratified_cross_validation(
            X, y, n_folds=10, random_state=self.random_state
        )
        oof_raw_probs = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_model = self.create_model(
                [
                    self.best_params["kernel"],
                    self.best_params["C"],
                    self.best_params["gamma"],
                    self.best_params["degree"],
                    self.best_params["coef0"],
                ]
            )
            fold_model.fit(X_train_fold, y_train_fold)
            oof_raw_probs[val_idx] = fold_model.predict_proba(X_val_fold)[:, 1]

        # Fit the same Platt (sigmoid) calibrator as for LR and RF
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
        ).to_csv("step_11_svm_calibration_parameters.csv", index=False)
        print(
            "SVM calibration parameters saved to: step_11_svm_calibration_parameters.csv"
        )

        oof_cal_probs = self.calibrator.predict_proba(oof_logits)[:, 1]

        # Global optimal thresholds from all calibrated OOF probabilities
        self.default = 0.5
        self.f1_median = float(f1_score_threshold(y, oof_cal_probs))
        self.youden_median = float(youden_threshold(y, oof_cal_probs))

        print("\n" + "=" * 60)
        print("Classification thresholds (from OOF predictions):")
        print("=" * 60)
        print(f"1. Default threshold:        0.5000")
        print(f"2. F1-optimal threshold:     {self.f1_median:.4f}")
        print(f"3. Youden-optimal threshold: {self.youden_median:.4f}")
        print("=" * 60)

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
            "Computing 10-fold CV performance on the training set (fixed global thresholds)..."
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
                    self.best_params["kernel"],
                    self.best_params["C"],
                    self.best_params["gamma"],
                    self.best_params["degree"],
                    self.best_params["coef0"],
                ]
            )
            fold_model.fit(X_train_fold, y_train_fold)
            y_val_raw_prob = fold_model.predict_proba(X_val_fold)[:, 1]
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
        Compute and plot feature importance:
          1. Weights (linear kernel only)
          2. SHAP (KernelExplainer, any kernel)
          3. Permutation importance

        Parameters
        ----
        X : np.ndarray
        y : np.ndarray
        feature_names : list of str
        """
        print("Computing SVM feature importance...")

        importances = {}

        # 1. Linear-kernel weights (absolute values)
        if self.model.kernel == "linear":
            importances["SVM_Weights"] = np.abs(self.model.coef_[0])

        # 2. SHAP: as for the other models, a fixed sample of at most 500 training participants.
        sample_size = min(500, len(X))
        sample_indices = np.random.RandomState(self.random_state).choice(
            len(X), sample_size, replace=False
        )
        X_sample = X[sample_indices]

        print(f"SVM SHAP: {sample_size} participants; {len(X)} background rows")
        explainer = shap.KernelExplainer(self.model.predict_proba, X)
        shap_values = explainer.shap_values(X_sample)
        # shap_values have shape (n_samples, n_features, n_classes); take the positive class (index 1)
        shap_values = shap_values[:, :, 1]
        importances["SHAP"] = np.mean(np.abs(shap_values), axis=0)
        plot_shap_summary(shap_values, X_sample, feature_names, "SVM")

        # 3. Permutation importance
        perm_importance = get_permutation_importance(
            self.model, X, y, feature_names, self.random_state
        )
        importances["Permutation"] = perm_importance

        plot_feature_importance_comparison(importances, feature_names, "SVM")

        return importances

    # ----------------------------------------------------------
    # Test-set performance
    # ----------------------------------------------------------

    def evaluate_and_save_results(self, X_test, y_test):
        """
        Compute metrics on the independent test set at the default, F1-optimal and Youden-optimal thresholds.

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


def restore_model_for_explanation(
    svm_model, X_train, y_train, X_test, y_test, test_patno
):
    """Refit saved parameters and verify existing predictions before writing explanations."""
    # Preserve the exact saved float: even a one-bit C change can alter SVC fitting.
    saved_params = pd.read_csv(
        "step_11_svm_best_hyperparameters.csv", float_precision="round_trip"
    )
    if len(saved_params) != 1 or saved_params.iloc[0]["model"] != "SVM":
        raise ValueError("Expected one saved SVM hyperparameter row")
    params = saved_params.iloc[0]
    gamma = params["gamma"]
    if gamma not in ("scale", "auto"):
        gamma = float(gamma)
    svm_model.model = svm_model.create_model(
        [
            params["kernel"],
            float(params["C"]),
            gamma,
            int(params["degree"]),
            float(params["coef0"]),
        ]
    )
    svm_model.model.fit(X_train, y_train)

    saved = pd.read_csv("step_11_svm_predictions.csv")
    if not (
        np.array_equal(saved["PATNO"].to_numpy(), np.asarray(test_patno))
        and np.array_equal(saved["y_true"].to_numpy(), y_test)
    ):
        raise ValueError(
            "Saved SVM predictions do not match the current test participants/labels"
        )
    restored = svm_model.model.predict_proba(X_test)[:, 1]
    if not np.allclose(restored, saved["y_prob_raw"].to_numpy(), rtol=0, atol=1e-10):
        raise ValueError(
            "Restored SVM differs from saved predictions; explanation outputs were not changed"
        )
    print("Saved-model prediction check passed; updating explanation outputs only.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train SVM or refresh its feature explanations"
    )
    parser.add_argument(
        "--explain-only",
        action="store_true",
        help="Reuse saved hyperparameters and verify predictions; refresh SHAP and feature importance only",
    )
    args = parser.parse_args(argv)
    np.random.seed(42)

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    train_data, test_data, feature_name_mapping = load_data(
        "PPMI_4_train.csv",
        "PPMI_4_test.csv",
        "PPMI_feature_mapping.csv",
        "step_09_selected_features.csv",
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
    feature_names = [
        feature_name_mapping.get(col, col)
        for col in train_data.columns
        if col not in ["PATNO", "Target"]
    ]

    # ----------------------------------------------------------
    # 3. Bayesian hyperparameter optimisation
    # ----------------------------------------------------------
    svm_model = SVMModel(random_state=42)
    if args.explain_only:
        restore_model_for_explanation(
            svm_model, X_train, y_train, X_test, y_test, test_patno
        )
        svm_model.evaluate_feature_importance(X_train, y_train, feature_names)
        return
    svm_model.optimize_hyperparameters(X_train, y_train)

    # ----------------------------------------------------------
    # 4. Final model and optimal classification thresholds
    # ----------------------------------------------------------
    svm_model.train_final_model(X_train, y_train)

    # ----------------------------------------------------------
    # 5. 10-fold CV performance on the training set
    # ----------------------------------------------------------
    train_performance = svm_model.get_cross_validation_performance(X_train, y_train)

    # ----------------------------------------------------------
    # 6. Feature importance
    # ----------------------------------------------------------
    svm_model.evaluate_feature_importance(X_train, y_train, feature_names)

    # ----------------------------------------------------------
    # 7. Test-set evaluation
    # ----------------------------------------------------------
    test_performance, y_prob_raw, y_prob = svm_model.evaluate_and_save_results(
        X_test, y_test
    )

    # Plot ROC and PR curves
    plot_roc_pr_curves(
        y_test,
        y_prob,
        "SVM",
        calibration_method="platt_10fold_oof",
        patient_ids=test_patno.to_numpy(),
        raw_prob=y_prob_raw,
        model_version="step_11_svm_oof_platt_v1",
    )

    # ----------------------------------------------------------
    # 8. Save and print results
    # ----------------------------------------------------------
    train_df = pd.DataFrame(train_performance)
    train_df.to_csv("step_11_svm_train_performance.csv")
    print(f"\nTraining-set performance saved to: step_11_svm_train_performance.csv")

    test_df = pd.DataFrame(test_performance)
    test_df.to_csv("step_11_svm_test_performance.csv")
    print(f"Test-set performance saved to: step_11_svm_test_performance.csv")

    print("\n" + "=" * 80)
    print("SVM training-set performance (mean ± SD):")
    print("=" * 80)
    print(train_df.round(4))

    print("\n" + "=" * 80)
    print("SVM test-set performance:")
    print("=" * 80)
    print(test_df.round(4))


if __name__ == "__main__":
    main()
