"""
Hiring Bias Fairness Analysis (adapted from the network fairness workflow).

This script reuses the same core approach:
1. Build an AIF360 BinaryLabelDataset
2. Train multiple ML models and evaluate fairness metrics
3. Apply Reweighing and retrain
4. Compare fairness/accuracy before vs after mitigation

Default dataset: D:\\Hiring _Bias\\stackoverflow_full.csv
Protected attribute: Gender (privileged = Man, unprivileged = not Man)
Label: Employed (1 = favorable, 0 = unfavorable)
"""

from __future__ import annotations

import argparse
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import aif360_local  # noqa: F401
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


DEFAULT_DATASET_PATH = Path(r"D:\Hiring _Bias\stackoverflow_full.csv")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fairness analysis on hiring dataset.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to input CSV dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30000,
        help="Optional downsample size for faster runs. Use 0 to use full dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def compute_fairness_metrics(
    dataset_true: BinaryLabelDataset,
    dataset_pred: BinaryLabelDataset,
    unprivileged_groups: list[dict[str, int]],
    privileged_groups: list[dict[str, int]],
) -> dict[str, float]:
    """Compute model fairness metrics with AIF360 ClassificationMetric."""
    cm = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    metrics: dict[str, float] = {}
    metrics["Balanced accuracy"] = 0.5 * (cm.true_positive_rate() + cm.true_negative_rate())
    metrics["Statistical parity difference"] = cm.statistical_parity_difference()
    metrics["Disparate impact"] = cm.disparate_impact()
    metrics["Average odds difference"] = cm.average_odds_difference()
    metrics["Equal opportunity difference"] = cm.equal_opportunity_difference()
    metrics["Theil index"] = cm.theil_index()
    return metrics


def fit_with_optional_weights(model, x_train: np.ndarray, y_train: np.ndarray, sample_weights=None):
    """Fit model and pass sample_weight when supported."""
    if sample_weights is None:
        model.fit(x_train, y_train)
        return

    fit_signature = inspect.signature(model.fit)
    if "sample_weight" in fit_signature.parameters:
        model.fit(x_train, y_train, sample_weight=sample_weights)
        return

    warnings.warn(
        f"{model.__class__.__name__} does not support sample_weight. Fitting without weights.",
        RuntimeWarning,
    )
    model.fit(x_train, y_train)


def prediction_scores(model, x_input: np.ndarray) -> np.ndarray:
    """Return a probability-like score in [0, 1] for threshold tuning."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_input)[:, 1]

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(x_input)
        min_score = np.min(raw_scores)
        max_score = np.max(raw_scores)
        if max_score == min_score:
            return np.full_like(raw_scores, 0.5, dtype=float)
        return (raw_scores - min_score) / (max_score - min_score)

    return model.predict(x_input).astype(float)


def train_and_evaluate(
    model,
    dataset_train: BinaryLabelDataset,
    dataset_valid: BinaryLabelDataset,
    dataset_test: BinaryLabelDataset,
    privileged_groups: list[dict[str, int]],
    unprivileged_groups: list[dict[str, int]],
    sample_weights=None,
) -> dict[str, float]:
    """Train model, tune threshold on validation, evaluate on test fairness + utility."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(dataset_train.features)
    y_train = dataset_train.labels.ravel().astype(int)

    fit_with_optional_weights(model, x_train, y_train, sample_weights=sample_weights)

    x_valid = scaler.transform(dataset_valid.features)
    y_valid = dataset_valid.labels.ravel().astype(int)
    valid_scores = prediction_scores(model, x_valid)

    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_bacc = -1.0

    for threshold in thresholds:
        valid_preds = (valid_scores >= threshold).astype(int)
        bacc = balanced_accuracy_score(y_valid, valid_preds)
        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = float(threshold)

    x_test = scaler.transform(dataset_test.features)
    y_test = dataset_test.labels.ravel().astype(int)
    test_scores = prediction_scores(model, x_test)
    y_pred = (test_scores >= best_threshold).astype(int)

    dataset_test_pred = dataset_test.copy(deepcopy=True)
    dataset_test_pred.labels = y_pred.reshape(-1, 1).astype(float)

    metrics = compute_fairness_metrics(
        dataset_true=dataset_test,
        dataset_pred=dataset_test_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_test, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_test, y_pred, zero_division=0)
    metrics["Threshold"] = best_threshold
    return metrics


def load_and_prepare_data(path: Path, sample_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV and return (original_df, analysis_df_for_aif360)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if sample_size and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    if "Gender" not in df.columns or "Employed" not in df.columns:
        raise ValueError("Dataset must include 'Gender' and 'Employed' columns.")

    # Protect group: privileged = Man, unprivileged = Woman/NonBinary/other.
    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Is_Man"] = (df["Gender"].astype(str).str.strip().str.lower() == "man").astype(int)

    # Keep a practical subset of features; 'HaveWorkedWith' is near-unique and not useful as one-hot.
    base_feature_cols = [
        "Age",
        "Accessibility",
        "EdLevel",
        "Employment",
        "MentalHealth",
        "MainBranch",
        "YearsCode",
        "YearsCodePro",
        "PreviousSalary",
        "ComputerSkills",
    ]

    missing_cols = [col for col in base_feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected feature columns: {missing_cols}")

    feature_df = df[base_feature_cols].copy()

    categorical_cols = ["Age", "Accessibility", "EdLevel", "MentalHealth", "MainBranch"]
    numeric_cols = ["Employment", "YearsCode", "YearsCodePro", "PreviousSalary", "ComputerSkills"]

    for col in categorical_cols:
        feature_df[col] = feature_df[col].fillna("Unknown").astype(str)

    for col in numeric_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    encoded_features = pd.get_dummies(
        feature_df,
        columns=categorical_cols,
        drop_first=True,
        dtype=float,
    )

    analysis_df = encoded_features.copy()
    analysis_df["Is_Man"] = df["Is_Man"].astype(float)

    labels = pd.to_numeric(df["Employed"], errors="coerce").fillna(0)
    labels = (labels > 0).astype(float)
    analysis_df["Employed"] = labels

    return df, analysis_df


def run_analysis(args: argparse.Namespace) -> None:
    print("=== Hiring Fairness Analysis (AIF360) ===")
    print(f"Dataset path: {args.dataset_path}")

    original_df, analysis_df = load_and_prepare_data(args.dataset_path, args.sample_size, args.random_state)

    print(f"Rows used: {len(original_df):,}")
    print(f"Features after encoding: {analysis_df.shape[1] - 2}")
    print("\nEmployed distribution:")
    print(original_df["Employed"].value_counts().sort_index())

    print("\nGender distribution:")
    print(original_df["Gender"].value_counts())

    group_rates = (
        pd.DataFrame(
            {
                "Is_Man": analysis_df["Is_Man"].astype(int),
                "Employed": analysis_df["Employed"].astype(int),
            }
        )
        .groupby("Is_Man")["Employed"]
        .mean()
        .rename(index={1: "Privileged (Man)", 0: "Unprivileged (Not Man)"})
    )
    print("\nEmployment rate by protected group:")
    print(group_rates)

    bld = BinaryLabelDataset(
        df=analysis_df,
        label_names=["Employed"],
        favorable_label=1.0,
        unfavorable_label=0.0,
        protected_attribute_names=["Is_Man"],
    )

    privileged_groups = [{"Is_Man": 1}]
    unprivileged_groups = [{"Is_Man": 0}]

    dataset_metric = BinaryLabelDatasetMetric(
        bld,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    print("\n=== Dataset-Level Fairness (Before Mitigation) ===")
    print(f"Mean difference (SPD): {dataset_metric.mean_difference():.4f}")
    print(f"Disparate impact: {dataset_metric.disparate_impact():.4f}")
    print(f"Base rate privileged: {dataset_metric.base_rate(privileged=True):.4f}")
    print(f"Base rate unprivileged: {dataset_metric.base_rate(privileged=False):.4f}")

    dataset_train, dataset_valid, dataset_test = bld.split([0.7, 0.85], shuffle=True, seed=args.random_state)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=args.random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=args.random_state),
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "Linear SVC": LinearSVC(random_state=args.random_state, dual="auto", max_iter=5000),
    }

    print("\n=== Model Results: BEFORE Reweighing ===")
    before_results: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        before_results[model_name] = train_and_evaluate(
            model=clone(model),
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            dataset_test=dataset_test,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            sample_weights=None,
        )

    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_rw_train = rw.fit_transform(dataset_train)

    train_metric_before = BinaryLabelDatasetMetric(
        dataset_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    train_metric_after = BinaryLabelDatasetMetric(
        dataset_rw_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    print("\n=== Dataset-Level Fairness on Train Split ===")
    print(
        f"SPD before: {train_metric_before.mean_difference():.4f} | "
        f"after: {train_metric_after.mean_difference():.4f}"
    )
    print(
        f"DI before: {train_metric_before.disparate_impact():.4f} | "
        f"after: {train_metric_after.disparate_impact():.4f}"
    )
    print(
        "Reweighing instance weight range: "
        f"{dataset_rw_train.instance_weights.min():.4f} - {dataset_rw_train.instance_weights.max():.4f}"
    )

    print("\n=== Model Results: AFTER Reweighing ===")
    after_results: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        after_results[model_name] = train_and_evaluate(
            model=clone(model),
            dataset_train=dataset_rw_train,
            dataset_valid=dataset_valid,
            dataset_test=dataset_test,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            sample_weights=dataset_rw_train.instance_weights,
        )

    before_df = pd.DataFrame(before_results).T
    after_df = pd.DataFrame(after_results).T

    summary_cols = [
        "Balanced accuracy",
        "Statistical parity difference",
        "Disparate impact",
        "Average odds difference",
        "Equal opportunity difference",
        "Accuracy",
        "F1",
        "Threshold",
    ]

    print("\n=== Summary: BEFORE Reweighing ===")
    print(before_df[summary_cols].round(4))

    print("\n=== Summary: AFTER Reweighing ===")
    print(after_df[summary_cols].round(4))

    comparison = pd.DataFrame(index=before_df.index)
    comparison["SPD_before"] = before_df["Statistical parity difference"]
    comparison["SPD_after"] = after_df["Statistical parity difference"]
    comparison["SPD_improvement_abs"] = comparison["SPD_before"].abs() - comparison["SPD_after"].abs()
    comparison["DI_before"] = before_df["Disparate impact"]
    comparison["DI_after"] = after_df["Disparate impact"]
    comparison["Accuracy_before"] = before_df["Accuracy"]
    comparison["Accuracy_after"] = after_df["Accuracy"]

    print("\n=== Change Summary (After - Before) ===")
    print(comparison.round(4))


if __name__ == "__main__":
    cli_args = parse_args()
    run_analysis(cli_args)
