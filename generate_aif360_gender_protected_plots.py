"""Generate before/after AIF360 plots for hiring fairness analysis.

Outputs:
- Model-level fairness/utility comparison (before vs after reweighing)
- Predicted hiring rates by Gender (before vs after) for the best baseline model
- Predicted hiring rates by MentalHealth (before vs after) for the best baseline model
- Predicted hiring rates by Age (before vs after) for the best baseline model
- Predicted hiring rates by Country (before vs after) for the best baseline model
- Interaction heatmaps (Gender x MentalHealth) before and after
- CSV summaries for downstream reporting
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
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate before/after AIF360 fairness plots.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--top-countries",
        type=int,
        default=15,
        help="Number of highest-support countries to show in the country bar chart.",
    )
    parser.add_argument(
        "--min-country-count",
        type=int,
        default=20,
        help="Minimum test-set sample size required for a country to appear in the chart.",
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
    """Train model, tune threshold on validation, return metrics and test predictions."""
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


def bar_before_after(
    ax,
    labels: list[str],
    before_values: np.ndarray,
    after_values: np.ndarray,
    title: str,
    ylabel: str,
    ref_line=None,
) -> None:
    x = np.arange(len(labels))
    width = 0.38

    ax.bar(x - width / 2, before_values, width=width, label="Before AIF360", color="#d95f02")
    ax.bar(x + width / 2, after_values, width=width, label="After AIF360", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    if ref_line is not None:
        ax.axhline(ref_line, color="black", linestyle="--", linewidth=1.0, alpha=0.7)


def save_model_comparison_plot(before_df: pd.DataFrame, after_df: pd.DataFrame, out_dir: Path) -> None:
    labels = before_df.index.tolist()
    spd_before = before_df["Statistical parity difference"].values
    spd_after = after_df["Statistical parity difference"].values
    di_before = before_df["Disparate impact"].values
    di_after = after_df["Disparate impact"].values
    acc_before = before_df["Accuracy"].values
    acc_after = after_df["Accuracy"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bar_before_after(
        axes[0],
        labels,
        spd_before,
        spd_after,
        title="Statistical Parity Difference by Model",
        ylabel="SPD (closer to 0 is fairer)",
        ref_line=0.0,
    )
    bar_before_after(
        axes[1],
        labels,
        di_before,
        di_after,
        title="Disparate Impact by Model",
        ylabel="DI (closer to 1 is fairer)",
        ref_line=1.0,
    )
    bar_before_after(
        axes[2],
        labels,
        acc_before,
        acc_after,
        title="Accuracy by Model",
        ylabel="Accuracy",
        ref_line=None,
    )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(out_dir / "aif360_model_metrics_before_after.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_group_rate_plot(
    df: pd.DataFrame,
    group_col: str,
    before_col: str,
    after_col: str,
    title: str,
    out_path: Path,
) -> None:
    grouped = df.groupby(group_col)[[before_col, after_col]].mean().sort_index()
    labels = grouped.index.astype(str).tolist()
    before_values = grouped[before_col].values
    after_values = grouped[after_col].values

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, before_values, width=width, label="Before AIF360", color="#7570b3")
    ax.bar(x + width / 2, after_values, width=width, label="After AIF360", color="#66a61e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted Hiring Rate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_country_rate_plot(
    pred_df: pd.DataFrame,
    out_path: Path,
    top_countries: int,
    min_country_count: int,
) -> pd.DataFrame:
    """Save before/after predicted hiring rates by country and return summary table."""
    country_summary = (
        pred_df.groupby("Country")[["Pred_Before", "Pred_After"]]
        .mean()
        .join(pred_df.groupby("Country").size().rename("n"))
        .sort_values("n", ascending=False)
    )

    filtered = country_summary[country_summary["n"] >= min_country_count].head(top_countries)
    if filtered.empty:
        filtered = country_summary.head(top_countries)

    labels = filtered.index.astype(str).tolist()
    before_values = filtered["Pred_Before"].values
    after_values = filtered["Pred_After"].values

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, before_values, width=width, label="Before AIF360", color="#7570b3")
    ax.bar(x + width / 2, after_values, width=width, label="After AIF360", color="#66a61e")

    ax.set_title("Predicted Hiring Rate by Country (Best Baseline Model)")
    ax.set_ylabel("Predicted Hiring Rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return filtered.reset_index().rename(columns={"index": "Country"})


def save_interaction_heatmaps(pred_df: pd.DataFrame, out_dir: Path) -> None:
    before_tab = pred_df.groupby(["Gender", "MentalHealth"])["Pred_Before"].mean().unstack(fill_value=np.nan)
    after_tab = pred_df.groupby(["Gender", "MentalHealth"])["Pred_After"].mean().unstack(fill_value=np.nan)

    common_gender = sorted(set(before_tab.index).union(after_tab.index))
    common_mental = sorted(set(before_tab.columns).union(after_tab.columns))
    before_tab = before_tab.reindex(index=common_gender, columns=common_mental)
    after_tab = after_tab.reindex(index=common_gender, columns=common_mental)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, table, title in [
        (axes[0], before_tab, "Before AIF360"),
        (axes[1], after_tab, "After AIF360"),
    ]:
        im = ax.imshow(table.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Predicted Hiring Rate Heatmap ({title})")
        ax.set_xlabel("MentalHealth")
        ax.set_ylabel("Gender")
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels(table.columns)
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels(table.index)

        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                val = table.values[r, c]
                txt = "NA" if np.isnan(val) else f"{val:.1%}"
                ax.text(c, r, txt, ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Predicted Hiring Rate")

    fig.savefig(out_dir / "aif360_gender_mental_health_heatmap_before_after.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    original_df, analysis_df = load_and_prepare_data(args.dataset_path, args.sample_size, args.random_state)

    bld = BinaryLabelDataset(
        df=analysis_df,
        label_names=["Employed"],
        favorable_label=1.0,
        unfavorable_label=0.0,
        protected_attribute_names=["Is_Man"],
    )
    privileged_groups = [{"Is_Man": 1}]
    unprivileged_groups = [{"Is_Man": 0}]

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
    for model_name, model in models.items():
        metrics, preds = train_eval_with_predictions(
            model=clone(model),
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            dataset_test=dataset_test,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            sample_weights=None,
        )
        before_results[model_name] = metrics
        before_preds[model_name] = preds

    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_rw_train = rw.fit_transform(dataset_train)

    after_results: dict[str, dict[str, float]] = {}
    after_preds: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        metrics, preds = train_eval_with_predictions(
            model=clone(model),
            dataset_train=dataset_rw_train,
            dataset_valid=dataset_valid,
            dataset_test=dataset_test,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            sample_weights=dataset_rw_train.instance_weights,
        )
        after_results[model_name] = metrics
        after_preds[model_name] = preds

    before_df = pd.DataFrame(before_results).T
    after_df = pd.DataFrame(after_results).T

    save_model_comparison_plot(before_df, after_df, args.out_dir)

    best_model = before_df["Balanced accuracy"].idxmax()
    test_indices = [int(i) for i in dataset_test.instance_names]
    group_cols = ["Gender", "MentalHealth", "Age"]
    if "Country" in original_df.columns:
        group_cols.append("Country")

    test_groups = original_df.loc[test_indices, group_cols].copy()
    test_groups["Gender"] = test_groups["Gender"].fillna("Unknown")
    test_groups["MentalHealth"] = test_groups["MentalHealth"].fillna("Unknown")
    test_groups["Age"] = test_groups["Age"].fillna("Unknown")
    if "Country" in test_groups.columns:
        test_groups["Country"] = test_groups["Country"].fillna("Unknown")

    pred_df = pd.DataFrame(
        {
            "Gender": test_groups["Gender"].astype(str).values,
            "MentalHealth": test_groups["MentalHealth"].astype(str).values,
            "Age": test_groups["Age"].astype(str).values,
            "Pred_Before": before_preds[best_model].astype(float),
            "Pred_After": after_preds[best_model].astype(float),
            "Actual": dataset_test.labels.ravel().astype(int),
        }
    )
    if "Country" in test_groups.columns:
        pred_df["Country"] = test_groups["Country"].astype(str).values

    save_group_rate_plot(
        pred_df,
        group_col="Gender",
        before_col="Pred_Before",
        after_col="Pred_After",
        title=f"Predicted Hiring Rate by Gender ({best_model})",
        out_path=args.out_dir / "aif360_pred_hiring_rate_by_gender_before_after.png",
    )
    save_group_rate_plot(
        pred_df,
        group_col="MentalHealth",
        before_col="Pred_Before",
        after_col="Pred_After",
        title=f"Predicted Hiring Rate by MentalHealth ({best_model})",
        out_path=args.out_dir / "aif360_pred_hiring_rate_by_mental_health_before_after.png",
    )
    save_group_rate_plot(
        pred_df,
        group_col="Age",
        before_col="Pred_Before",
        after_col="Pred_After",
        title=f"Predicted Hiring Rate by Age ({best_model})",
        out_path=args.out_dir / "aif360_pred_hiring_rate_by_age_before_after.png",
    )
    save_interaction_heatmaps(pred_df, args.out_dir)
    if "Country" in pred_df.columns:
        country_summary = save_country_rate_plot(
            pred_df=pred_df,
            out_path=args.out_dir / "aif360_pred_hiring_rate_by_country_before_after.png",
            top_countries=args.top_countries,
            min_country_count=args.min_country_count,
        )
        country_summary.to_csv(
            args.out_dir / "aif360_pred_rate_by_country_before_after.csv",
            index=False,
        )

    before_df.to_csv(args.out_dir / "aif360_model_metrics_before.csv")
    after_df.to_csv(args.out_dir / "aif360_model_metrics_after.csv")
    pred_df.groupby("Gender")[["Pred_Before", "Pred_After"]].mean().to_csv(
        args.out_dir / "aif360_pred_rate_by_gender_before_after.csv"
    )
    pred_df.groupby("MentalHealth")[["Pred_Before", "Pred_After"]].mean().to_csv(
        args.out_dir / "aif360_pred_rate_by_mental_health_before_after.csv"
    )
    pred_df.groupby("Age")[["Pred_Before", "Pred_After"]].mean().to_csv(
        args.out_dir / "aif360_pred_rate_by_age_before_after.csv"
    )
    pred_df.groupby(["Gender", "MentalHealth"])[["Pred_Before", "Pred_After"]].mean().to_csv(
        args.out_dir / "aif360_pred_rate_by_gender_mental_before_after.csv"
    )
    if "Country" in pred_df.columns:
        pred_df.groupby("Country")[["Pred_Before", "Pred_After"]].mean().to_csv(
            args.out_dir / "aif360_pred_rate_by_country_all_before_after.csv"
        )

    print(f"Saved AIF360 before/after plots to: {args.out_dir}")
    print(f"Best baseline model for group plots: {best_model}")


if __name__ == "__main__":
    main()


