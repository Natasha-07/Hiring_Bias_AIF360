"""Generate before/after AIF360 results with Country as the protected attribute.

Because country has many categories, this script uses one-vs-rest protection:
- privileged group: Country == privileged_country (default: United States of America)
- unprivileged group: all other countries
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import aif360_local  # noqa: F401
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.datasets import BinaryLabelDataset
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

from hiring_bias_fairness_analysis import (
    compute_fairness_metrics,
    fit_with_optional_weights,
    load_and_prepare_data,
    prediction_scores,
)


DEFAULT_DATASET = Path(r"D:\Hiring _Bias\stackoverflow_full.csv")
DEFAULT_OUT_DIR = Path(r"D:\Hiring _Bias\plots")
DEFAULT_PRIVILEGED_COUNTRY = "United States of America"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate country-protected AIF360 before/after plots.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--privileged-country",
        type=str,
        default=DEFAULT_PRIVILEGED_COUNTRY,
        help="Country treated as privileged in one-vs-rest fairness setup.",
    )
    parser.add_argument(
        "--top-countries",
        type=int,
        default=15,
        help="Number of top countries (by test sample size) to show in supplemental plot.",
    )
    parser.add_argument(
        "--min-country-count",
        type=int,
        default=20,
        help="Minimum test-set count required for country in supplemental plot.",
    )
    return parser.parse_args()


def train_eval_with_predictions(
    model,
    dataset_train: BinaryLabelDataset,
    dataset_valid: BinaryLabelDataset,
    dataset_test: BinaryLabelDataset,
    privileged_groups: list[dict[str, int]],
    unprivileged_groups: list[dict[str, int]],
    sample_weights=None,
) -> tuple[dict[str, float], np.ndarray]:
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
    return metrics, y_pred


def save_model_comparison_plot(before_df: pd.DataFrame, after_df: pd.DataFrame, out_dir: Path, privileged_country: str) -> None:
    labels = before_df.index.tolist()
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("Statistical parity difference", "SPD (closer to 0 is fairer)", 0.0),
        ("Disparate impact", "DI (closer to 1 is fairer)", 1.0),
        ("Accuracy", "Accuracy", None),
    ]

    for ax, (metric_name, ylabel, ref_line) in zip(axes, metrics):
        before_vals = before_df[metric_name].values
        after_vals = after_df[metric_name].values
        ax.bar(x - width / 2, before_vals, width=width, label="Before AIF360", color="#d95f02")
        ax.bar(x + width / 2, after_vals, width=width, label="After AIF360", color="#1b9e77")
        if ref_line is not None:
            ax.axhline(ref_line, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(metric_name)
        ax.grid(axis="y", alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Country-Protected AIF360 (Privileged: {privileged_country})", y=1.08, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "aif360_country_protected_model_metrics_before_after.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_country_group_plot(pred_df: pd.DataFrame, out_dir: Path, model_name: str, privileged_country: str) -> None:
    grouped = pred_df.groupby("Country_Group")[["Pred_Before", "Pred_After"]].mean()
    order = [privileged_country, "Other Countries"]
    grouped = grouped.reindex([grp for grp in order if grp in grouped.index])

    labels = grouped.index.tolist()
    before_vals = grouped["Pred_Before"].values
    after_vals = grouped["Pred_After"].values

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before_vals, width=width, label="Before AIF360", color="#7570b3")
    ax.bar(x + width / 2, after_vals, width=width, label="After AIF360", color="#66a61e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted Hiring Rate")
    ax.set_title(f"Predicted Hiring Rate by Country Group ({model_name})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        out_dir / "aif360_country_protected_pred_hiring_rate_by_country_group_before_after.png",
        dpi=180,
    )
    plt.close(fig)


def save_top_country_plot(
    pred_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
    top_countries: int,
    min_country_count: int,
) -> None:
    summary = (
        pred_df.groupby("Country")[["Pred_Before", "Pred_After"]]
        .mean()
        .join(pred_df.groupby("Country").size().rename("n"))
        .sort_values("n", ascending=False)
    )
    filtered = summary[summary["n"] >= min_country_count].head(top_countries)
    if filtered.empty:
        filtered = summary.head(top_countries)

    labels = filtered.index.tolist()
    before_vals = filtered["Pred_Before"].values
    after_vals = filtered["Pred_After"].values

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6))
    ax.bar(x - width / 2, before_vals, width=width, label="Before AIF360", color="#7570b3")
    ax.bar(x + width / 2, after_vals, width=width, label="After AIF360", color="#66a61e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted Hiring Rate")
    ax.set_title(f"Top Countries Predicted Hiring Rate ({model_name})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "aif360_country_protected_pred_hiring_rate_top_countries_before_after.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    original_df, analysis_df = load_and_prepare_data(args.dataset_path, args.sample_size, args.random_state)
    country_raw = original_df["Country"].fillna("Unknown").astype(str).str.strip()

    # privileged = selected country, unprivileged = all others
    analysis_df = analysis_df.copy()
    analysis_df["Is_Priv_Country"] = (country_raw == args.privileged_country).astype(float)

    bld = BinaryLabelDataset(
        df=analysis_df,
        label_names=["Employed"],
        favorable_label=1.0,
        unfavorable_label=0.0,
        protected_attribute_names=["Is_Priv_Country"],
    )
    privileged_groups = [{"Is_Priv_Country": 1}]
    unprivileged_groups = [{"Is_Priv_Country": 0}]

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

    before_results: dict[str, dict[str, float]] = {}
    before_preds: dict[str, np.ndarray] = {}
    for name, model in models.items():
        metrics, preds = train_eval_with_predictions(
            model=clone(model),
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            dataset_test=dataset_test,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            sample_weights=None,
        )
        before_results[name] = metrics
        before_preds[name] = preds

    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_rw_train = rw.fit_transform(dataset_train)

    after_results: dict[str, dict[str, float]] = {}
    after_preds: dict[str, np.ndarray] = {}
    for name, model in models.items():
        metrics, preds = train_eval_with_predictions(
            model=clone(model),
            dataset_train=dataset_rw_train,
            dataset_valid=dataset_valid,
            dataset_test=dataset_test,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            sample_weights=dataset_rw_train.instance_weights,
        )
        after_results[name] = metrics
        after_preds[name] = preds

    before_df = pd.DataFrame(before_results).T
    after_df = pd.DataFrame(after_results).T
    save_model_comparison_plot(before_df, after_df, args.out_dir, args.privileged_country)

    best_model = before_df["Balanced accuracy"].idxmax()
    test_indices = [int(i) for i in dataset_test.instance_names]
    country_test = country_raw.iloc[test_indices].values
    pred_df = pd.DataFrame(
        {
            "Country": country_test,
            "Country_Group": np.where(country_test == args.privileged_country, args.privileged_country, "Other Countries"),
            "Pred_Before": before_preds[best_model].astype(float),
            "Pred_After": after_preds[best_model].astype(float),
            "Actual": dataset_test.labels.ravel().astype(int),
        }
    )
    save_country_group_plot(pred_df, args.out_dir, best_model, args.privileged_country)
    save_top_country_plot(
        pred_df,
        args.out_dir,
        best_model,
        top_countries=args.top_countries,
        min_country_count=args.min_country_count,
    )

    before_df.to_csv(args.out_dir / "aif360_country_protected_model_metrics_before.csv")
    after_df.to_csv(args.out_dir / "aif360_country_protected_model_metrics_after.csv")
    combined = before_df.add_suffix("_before").join(after_df.add_suffix("_after"))
    combined.to_csv(args.out_dir / "aif360_country_protected_all_models_before_after.csv")

    pred_df.groupby("Country_Group")[["Pred_Before", "Pred_After"]].mean().to_csv(
        args.out_dir / "aif360_country_protected_pred_rate_by_country_group_before_after.csv"
    )
    pred_df.groupby("Country_Group").size().rename("n").to_csv(
        args.out_dir / "aif360_country_protected_test_counts_by_country_group.csv"
    )
    pred_df.groupby("Country")[["Pred_Before", "Pred_After"]].mean().to_csv(
        args.out_dir / "aif360_country_protected_pred_rate_by_country_all_before_after.csv"
    )

    print(f"Saved country-protected AIF360 outputs to: {args.out_dir}")
    print(f"Privileged country: {args.privileged_country}")
    print(f"Best baseline model for country plots: {best_model}")


if __name__ == "__main__":
    main()


