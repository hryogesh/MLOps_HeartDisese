from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_dataset


def generate_eda_plots(output_dir: str | None = None) -> None:
    output_dir = Path(output_dir or "artifacts/eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(path="data/heart_disease.csv")

    plt.figure(figsize=(8, 5))
    df["target"].value_counts().plot(kind="bar", color=["#4C78A8", "#F58518"])
    plt.title("Class Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    df["age"].hist(bins=20, color="#54A24B")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png")
    plt.close()

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing values: {int(df.isnull().sum().sum())}\n"
    )


if __name__ == "__main__":
    generate_eda_plots()
