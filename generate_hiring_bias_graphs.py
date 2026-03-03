"""Generate bias/fairness charts for hiring outcomes.

Focus:
- Gender vs Employed (hiring rate and counts)
- MentalHealth vs Employed
- Interaction of Gender x MentalHealth vs Employed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(r"D:\Hiring _Bias\stackoverflow_full.csv")
DEFAULT_OUT_DIR = Path(r"D:\Hiring _Bias\plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hiring bias plots.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Gender", "MentalHealth", "Employed"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    out = df.copy()
    out["Gender"] = out["Gender"].fillna("Unknown").astype(str).str.strip()
    out["MentalHealth"] = out["MentalHealth"].fillna("Unknown").astype(str).str.strip()

    employed = pd.to_numeric(out["Employed"], errors="coerce").fillna(0)
    out["Employed"] = (employed > 0).astype(int)
    return out


def add_percentage_labels(ax, values: np.ndarray, fmt: str = "{:.1%}") -> None:
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, fmt.format(v), ha="center", va="bottom", fontsize=9)


def save_plot_hiring_rate_by_gender(df: pd.DataFrame, out_dir: Path) -> None:
    rates = df.groupby("Gender", sort=False)["Employed"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(rates.index, rates.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Hiring Rate by Gender")
    ax.set_ylabel("Hiring Rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(out_dir / "hiring_rate_by_gender.png", dpi=180)
    plt.close(fig)


def save_plot_hiring_rate_by_mental_health(df: pd.DataFrame, out_dir: Path) -> None:
    rates = df.groupby("MentalHealth", sort=False)["Employed"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(rates.index, rates.values, color=["#4daf4a", "#e41a1c", "#377eb8"])
    ax.set_title("Hiring Rate by MentalHealth Status")
    ax.set_ylabel("Hiring Rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(out_dir / "hiring_rate_by_mental_health.png", dpi=180)
    plt.close(fig)


def save_plot_grouped_gender_mental_health(df: pd.DataFrame, out_dir: Path) -> None:
    rate_table = (
        df.groupby(["Gender", "MentalHealth"])["Employed"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    genders = rate_table.index.tolist()
    mh_groups = rate_table.columns.tolist()
    x = np.arange(len(genders))
    width = 0.35 if len(mh_groups) <= 2 else max(0.15, 0.8 / len(mh_groups))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mh in enumerate(mh_groups):
        values = rate_table[mh].values
        ax.bar(x + (i - (len(mh_groups) - 1) / 2) * width, values, width=width, label=mh)

    ax.set_title("Hiring Rate by Gender and MentalHealth")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Hiring Rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(genders)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="MentalHealth")

    plt.tight_layout()
    fig.savefig(out_dir / "hiring_rate_by_gender_and_mental_health.png", dpi=180)
    plt.close(fig)


def save_plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    rate_table = (
        df.groupby(["Gender", "MentalHealth"])["Employed"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rate_table.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Hiring Rate Heatmap (Gender x MentalHealth)")
    ax.set_xlabel("MentalHealth")
    ax.set_ylabel("Gender")
    ax.set_xticks(np.arange(len(rate_table.columns)))
    ax.set_xticklabels(rate_table.columns)
    ax.set_yticks(np.arange(len(rate_table.index)))
    ax.set_yticklabels(rate_table.index)

    for row in range(rate_table.shape[0]):
        for col in range(rate_table.shape[1]):
            val = rate_table.values[row, col]
            label = "NA" if np.isnan(val) else f"{val:.1%}"
            ax.text(col, row, label, ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Hiring Rate")

    plt.tight_layout()
    fig.savefig(out_dir / "hiring_rate_heatmap_gender_mental_health.png", dpi=180)
    plt.close(fig)


def save_plot_counts(df: pd.DataFrame, out_dir: Path) -> None:
    count_table = (
        df.groupby(["Gender", "MentalHealth"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(count_table.index), dtype=float)

    colors = ["#8dd3c7", "#fb8072", "#80b1d3", "#fdb462"]
    for i, mh in enumerate(count_table.columns):
        vals = count_table[mh].values
        ax.bar(count_table.index, vals, bottom=bottom, label=mh, color=colors[i % len(colors)])
        bottom += vals

    ax.set_title("Sample Counts by Gender and MentalHealth")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="MentalHealth")

    plt.tight_layout()
    fig.savefig(out_dir / "sample_counts_by_gender_and_mental_health.png", dpi=180)
    plt.close(fig)


def save_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    rates_gender = (
        df.groupby("Gender")["Employed"]
        .agg(hiring_rate="mean", n="count")
        .sort_values("hiring_rate", ascending=False)
    )
    rates_mental = (
        df.groupby("MentalHealth")["Employed"]
        .agg(hiring_rate="mean", n="count")
        .sort_values("hiring_rate", ascending=False)
    )
    rates_interaction = (
        df.groupby(["Gender", "MentalHealth"])["Employed"]
        .agg(hiring_rate="mean", n="count")
        .reset_index()
        .sort_values(["Gender", "MentalHealth"])
    )

    rates_gender.to_csv(out_dir / "summary_hiring_rate_by_gender.csv")
    rates_mental.to_csv(out_dir / "summary_hiring_rate_by_mental_health.csv")
    rates_interaction.to_csv(out_dir / "summary_hiring_rate_by_gender_mental_health.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset_path)
    df = clean_dataframe(df)

    save_plot_hiring_rate_by_gender(df, args.out_dir)
    save_plot_hiring_rate_by_mental_health(df, args.out_dir)
    save_plot_grouped_gender_mental_health(df, args.out_dir)
    save_plot_heatmap(df, args.out_dir)
    save_plot_counts(df, args.out_dir)
    save_summary_tables(df, args.out_dir)

    print(f"Saved plots and summaries to: {args.out_dir}")


if __name__ == "__main__":
    main()
