from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_dataset
from src.models.model_comparison import compare_models


def generate_eda_plots(output_dir: str | None = None) -> None:
    """Generate comprehensive EDA plots and summary report."""
    output_dir = Path(output_dir or "artifacts/eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(path="data/heart_disease.csv")

    # ============================================
    # 1. BASIC DATA OVERVIEW & SUMMARY
    # ============================================
    print("=" * 60)
    print("1. BASIC DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"Total Missing: {int(df.isnull().sum().sum())}")

    # ============================================
    # 2. CLASS DISTRIBUTION
    # ============================================
    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION")
    print("=" * 60)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="target", palette=["#4C78A8", "#F58518"])
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "01_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Class distribution plot saved")

    # ============================================
    # 3. NUMERIC FEATURE DISTRIBUTIONS
    # ============================================
    print("\n" + "=" * 60)
    print("3. NUMERIC FEATURE DISTRIBUTIONS")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    print(f"Found {len(numeric_cols)} numeric features: {numeric_cols}")

    for i, col in enumerate(numeric_cols):
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=20, color="#54A24B")
        plt.title(f"{col} Distribution", fontsize=14, fontweight="bold")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / f"02_hist_{col}.png", dpi=300, bbox_inches="tight")
        plt.close()
    print(f"✓ Generated histograms for {len(numeric_cols)} features")

    # ============================================
    # 4. CORRELATION HEATMAP
    # ============================================
    print("\n" + "=" * 60)
    print("4. CORRELATION HEATMAP")
    print("=" * 60)
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, square=True, cbar_kws={"label": "Correlation"})
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "03_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Correlation heatmap saved")

    # ============================================
    # 5. FEATURE RELATIONSHIP WITH TARGET
    # ============================================
    print("\n" + "=" * 60)
    print("5. FEATURE RELATIONSHIP WITH TARGET")
    print("=" * 60)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="target", y="thalach", palette=["#4C78A8", "#F58518"])
    plt.title("Target vs Max Heart Rate (thalach)", fontsize=14, fontweight="bold")
    plt.xlabel("Target")
    plt.ylabel("Max Heart Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "04_target_vs_thalach.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Feature-target relationship plot saved")

    # ============================================
    # 6. MODEL COMPARISON & SELECTION
    # ============================================
    print("\n" + "=" * 60)
    print("6. MODEL COMPARISON & SELECTION")
    print("=" * 60)
    comparison = compare_models(df)
    print("\nModel Comparison Summary:")
    print("-" * 60)
    for name, metrics in comparison["results"].items():
        print(f"\n{name}:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
        print(f"  CV Accuracy:     {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")

    print(f"\n✓ Selected Model: {comparison['best_model']}")
    print(f"✓ Note: {comparison['tuning_note']}")

    # ============================================
    # 7. GENERATE SUMMARY REPORT
    # ============================================
    print("\n" + "=" * 60)
    print("7. GENERATING SUMMARY REPORT")
    print("=" * 60)

    summary_text = f"""
EXPLORATORY DATA ANALYSIS (EDA) REPORT
=====================================

1. DATASET OVERVIEW
-------------------
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing Values: {int(df.isnull().sum().sum())}

Features with missing values:
{df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().any() else 'None'}

2. DATA TYPES
-------------
{df.dtypes.to_string()}

3. CLASS DISTRIBUTION
---------------------
{df['target'].value_counts().to_string()}
Class Balance: {df['target'].value_counts(normalize=True).round(3).to_string()}

4. NUMERIC FEATURES SUMMARY
----------------------------
{df[numeric_cols].describe().to_string()}

5. CORRELATION INSIGHTS
------------------------
Numeric features analyzed: {len(numeric_cols)}
Features: {', '.join(numeric_cols)}

6. MODEL COMPARISON RESULTS
----------------------------
"""

    for name, metrics in comparison["results"].items():
        summary_text += f"""
{name}:
  - Accuracy:        {metrics['accuracy']:.4f}
  - Precision:       {metrics['precision']:.4f}
  - Recall:          {metrics['recall']:.4f}
  - ROC-AUC:         {metrics['roc_auc']:.4f}
  - CV Accuracy:     {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}
"""

    summary_text += f"""
Selected Model: {comparison['best_model']}
Note: {comparison['tuning_note']}

PLOTS GENERATED
---------------
1. 01_class_distribution.png - Class balance visualization
2. 02_hist_*.png - Individual feature distributions
3. 03_correlation_heatmap.png - Feature correlations
4. 04_target_vs_thalach.png - Target relationship analysis

EDA Complete ✓
"""

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary_text)
    print(f"✓ Summary report saved to {summary_path}")

    print("\n" + "=" * 60)
    print("EDA GENERATION COMPLETE")
    print("=" * 60)
    print(f"All outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    generate_eda_plots()
