"""Shared helpers for the step 11 classifiers: data loading, cross-validation,
thresholds, metrics and plots.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend, avoids crashes in native GUI libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score,
)
from sklearn.inspection import permutation_importance
import shap
import warnings

warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "default", category=ConvergenceWarning
)  # keep convergence warnings visible

# Global font settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = True


from sklearn.preprocessing import StandardScaler


# ============================================================
# Data loading and preprocessing
# ============================================================


def load_data(
    train_file="PPMI_4_train.csv",
    test_file="PPMI_4_test.csv",
    feature_mapping_file="PPMI_feature_mapping.csv",
    selected_features_file="step_09_selected_features.csv",
    return_scaler=False,
):
    """
    Load the training and test sets, keep the features listed in selected_features_file
    and z-standardise them with a StandardScaler fitted on the training set only.

    Parameters
    ----
    train_file : str
        Training-set CSV path (default 'PPMI_4_train.csv').
    test_file : str
        Test-set CSV path (default 'PPMI_4_test.csv').
    feature_mapping_file : str
        Feature-mapping CSV path with the columns 'Feature' and 'Abbreviation'.
    selected_features_file : str, optional
        CSV path of the final feature list (default 'step_09_selected_features.csv').
    return_scaler : bool, optional
        Also return the StandardScaler fitted on the training set (default False).

    Returns
    ----
    train_data : pd.DataFrame (PATNO, standardised final features, Target)
    test_data : pd.DataFrame (PATNO, standardised final features, Target)
    feature_name_mapping : dict (original feature name -> abbreviation)
    scaler : sklearn.preprocessing.StandardScaler (only if return_scaler=True)
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    feature_mapping = pd.read_csv(feature_mapping_file)
    feature_name_mapping = dict(
        zip(feature_mapping["Feature"], feature_mapping["Abbreviation"])
    )

    if selected_features_file is not None:
        sel_df = pd.read_csv(selected_features_file)
        feat_col = "feature" if "feature" in sel_df.columns else sel_df.columns[0]
        selected_features = sel_df[feat_col].dropna().tolist()
    else:
        selected_features = [
            c for c in train_df.columns if c not in ["PATNO", "Target"]
        ]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[selected_features]), columns=selected_features
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(test_df[selected_features]), columns=selected_features
    )

    train_data = X_train_scaled.copy()
    train_data.insert(0, "PATNO", train_df["PATNO"].values)
    train_data["Target"] = train_df["Target"].values

    test_data = X_test_scaled.copy()
    test_data.insert(0, "PATNO", test_df["PATNO"].values)
    test_data["Target"] = test_df["Target"].values

    if return_scaler:
        return train_data, test_data, feature_name_mapping, scaler
    return train_data, test_data, feature_name_mapping


def prepare_data(data, subject_col="PATNO", label_col="Target"):
    """
    Separate features, labels and participant IDs.

    Parameters
    ----
    data : pd.DataFrame
    subject_col : str, default 'PATNO'
        Name of the participant ID column.
    label_col : str, default 'Target'
        Name of the label column.

    Returns
    ----
    X : pd.DataFrame  feature matrix
    y : pd.Series     labels
    subjects : pd.Series  participant IDs
    """
    X = data.drop([subject_col, label_col], axis=1)
    y = data[label_col]
    subjects = data[subject_col]
    return X, y, subjects


# ============================================================
# Cross-validation
# ============================================================


def stratified_cross_validation(X, y, n_folds=10, random_state=42):
    """
    Standard stratified K-fold cross-validation.

    Suitable when each participant has a single record (e.g. baseline visits only),
    so no participant-level grouping is needed.

    Parameters
    ----
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    n_folds : int, default 10
    random_state : int, default 42

    Returns
    ----
    folds : list of (train_idx, val_idx) tuples
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = [(train_idx, val_idx) for train_idx, val_idx in skf.split(X, y)]
    return folds


# ============================================================
# Thresholds
# ============================================================


def f1_score_threshold(y_true, y_prob):
    """
    Threshold that maximises the F1 score along the precision-recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)[
        :-1
    ]  # drop the last point, which has no threshold
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


def youden_threshold(y_true, y_prob):
    """
    Threshold that maximises the Youden index (J = TPR - FPR) along the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_scores = tpr - fpr
    optimal_idx = np.argmax(youden_scores)
    return thresholds[optimal_idx]


# ============================================================
# Evaluation metrics
# ============================================================


def calculate_all_metrics(y_true, y_pred, y_prob, threshold=None):
    """
    Compute the full set of binary classification metrics.

    Returned fields (in fixed order): AUC-ROC, AUC-PR, Optimal Threshold,
    Accuracy, Balanced Accuracy, Precision, Recall, Specificity,
    F1-score, Cohen's Kappa.
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    result = {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "Optimal Threshold": threshold if threshold is not None else 0.5,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "Cohen's Kappa": cohen_kappa,
    }
    return result


# ============================================================
# Plotting
# ============================================================


def plot_roc_pr_curves(
    y_true,
    y_prob,
    model_name,
    save_prefix="step_11_",
    calibration_method=None,
    patient_ids=None,
    raw_prob=None,
    split="test",
    model_version=None,
):
    """
    Plot and save the ROC and PR curves.
    Also save the predicted probabilities as CSV.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    ax1.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_roc:.4f})", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title(f"{model_name} ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- PR curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    ax2.plot(recall, precision, label=f"{model_name} (AUC = {auc_pr:.4f})", linewidth=2)
    ax2.axhline(
        y=np.mean(y_true),
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline ({np.mean(y_true):.3f})",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{model_name} Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    m_name = model_name.lower()
    plt.savefig(f"{save_prefix}{m_name}_roc_pr_curves.pdf", bbox_inches="tight")
    plt.close()

    # Save raw and calibrated probabilities per patient for checking
    if patient_ids is None or raw_prob is None or model_version is None:
        raise ValueError(
            "patient_ids, raw_prob, and model_version are required when "
            "saving prediction outputs"
        )
    lengths = {len(y_true), len(y_prob), len(patient_ids), len(raw_prob)}
    if len(lengths) != 1:
        raise ValueError("Prediction output columns must have equal lengths")
    if pd.Series(patient_ids).isna().any() or pd.Series(patient_ids).duplicated().any():
        raise ValueError("Prediction outputs require unique, non-missing PATNO values")

    predictions_data = pd.DataFrame(
        {
            "PATNO": patient_ids,
            "split": split,
            "model_version": model_version,
            "y_true": y_true,
            "y_prob_raw": raw_prob,
            "y_prob": y_prob,
        }
    )
    if calibration_method is not None:
        predictions_data["calibration_method"] = calibration_method
    predictions_data.to_csv(f"{save_prefix}{m_name}_predictions.csv", index=False)


def plot_feature_importance_comparison(
    importances_dict, feature_names, model_name, save_prefix="step_11_"
):
    """
    Plot horizontal bar charts comparing the feature-importance methods (PDF).
    Also save the importance values of each method as CSV.
    """
    n_methods = len(importances_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8))

    if n_methods == 1:
        axes = [axes]

    for idx, (method_name, importance_values) in enumerate(importances_dict.items()):
        sorted_idx = np.argsort(np.abs(importance_values))
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = importance_values[sorted_idx]

        bars = axes[idx].barh(range(len(sorted_values)), sorted_values)
        axes[idx].set_yticks(range(len(sorted_values)))
        axes[idx].set_yticklabels(sorted_features)
        axes[idx].set_xlabel("Importance")
        axes[idx].set_title(f"{model_name} - {method_name}")
        axes[idx].grid(True, alpha=0.3)

        # Negative values in red, positive in blue
        for i, bar in enumerate(bars):
            bar.set_color("#FF0051" if sorted_values[i] < 0 else "#008BFB")

    plt.tight_layout()
    m_name = model_name.lower()
    plt.savefig(f"{save_prefix}{m_name}_feature_importance.pdf", bbox_inches="tight")
    plt.close()

    # Save the feature-importance values
    importance_df = pd.DataFrame(importances_dict, index=feature_names)
    importance_df.to_csv(f"{save_prefix}{m_name}_feature_importance.csv")


def plot_shap_summary(
    shap_values, X, feature_names, model_name, save_prefix="step_11_"
):
    """
    Plot and save the SHAP bar and beeswarm summaries (PDF).
    Also save the SHAP values as CSV.
    """
    m_name = model_name.lower()
    # SHAP bar chart (mean absolute value)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.title(f"{model_name} - SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}{m_name}_shap_bar.pdf", bbox_inches="tight")
    plt.close()

    # SHAP beeswarm summary (direction and distribution)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f"{model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}{m_name}_shap_summary.pdf", bbox_inches="tight")
    plt.close()

    # Save the SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(f"{save_prefix}{m_name}_shap_values.csv", index=False)


# ============================================================
# Other helpers
# ============================================================


def get_permutation_importance(model, X, y, feature_names, random_state=42):
    """
    Permutation importance (AUC-ROC scoring, mean of 10 repeats).
    """
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=random_state, scoring="roc_auc"
    )
    return perm_importance.importances_mean


def print_results(model_name, metrics, optimal_threshold):
    """Print a summary of the evaluation results."""
    print(f"\n{'=' * 50}")
    print(f"{model_name} evaluation results")
    print(f"{'=' * 50}")
    print(f"Optimal classification threshold: {optimal_threshold:.4f}")

    print(f"\nConfusion matrix:")
    print(metrics["Confusion Matrix"])

    print("\nMetrics:")
    for metric, value in metrics.items():
        if metric != "Confusion Matrix":
            print(f"  {metric}: {value}")
