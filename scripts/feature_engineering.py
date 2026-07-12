"""
Feature Engineering Script
Handles feature selection and preprocessing for the heart disease dataset.

Usage:
    python scripts/feature_engineering.py --mode selection
    python scripts/feature_engineering.py --mode preprocessing
    python scripts/feature_engineering.py --mode all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_dataset
from src.features.feature_selection import select_features
from src.features.preprocess import prepare_features


def run_feature_selection(output_dir: str | None = None) -> None:
    """Run feature selection and save results."""
    output_dir = Path(output_dir or "artifacts/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    # Load data
    df = load_dataset(path="data/heart_disease.csv")
    print(f"\n✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Run feature selection
    print("\nSelecting top 8 features using ANOVA F-statistic...")
    result = select_features(df, k=8)

    selected_features = result["selected_columns"]
    print(f"✓ Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")

    # Get feature scores
    selector = result["selector"]
    all_features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    scores = selector.scores_

    print("\nFeature Importance Scores:")
    print("-" * 70)
    feature_scores = sorted(zip(all_features, scores), key=lambda x: x[1], reverse=True)
    for feat, score in feature_scores:
        marker = "✓ SELECTED" if feat in selected_features else ""
        print(f"  {feat:15} → {score:8.2f}  {marker}")

    # Save results
    results_file = output_dir / "feature_selection_results.txt"
    with open(results_file, "w") as f:
        f.write("FEATURE SELECTION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Selected Features ({len(selected_features)}):\n")
        for feat in selected_features:
            f.write(f"  - {feat}\n")
        f.write("\n\nFeature Importance Scores:\n")
        for feat, score in feature_scores:
            f.write(f"  {feat}: {score:.4f}\n")

    print(f"\n✓ Results saved to {results_file}")


def run_preprocessing(output_dir: str | None = None) -> None:
    """Run feature preprocessing and save statistics."""
    output_dir = Path(output_dir or "artifacts/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("FEATURE PREPROCESSING")
    print("=" * 70)

    # Load data
    df = load_dataset(path="data/heart_disease.csv")
    print(f"\n✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Run preprocessing
    print("\nPreprocessing features...")
    print("  - Imputing missing values (median strategy)")
    print("  - Scaling features (StandardScaler)")
    print("  - Splitting data (80% train / 20% test, stratified)")

    prepared = prepare_features(df)

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    print(f"\n✓ Preprocessing complete!")
    print(f"\nData Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    print(f"\nClass Distribution (Train):")
    y_train_counts = pd.Series(y_train).value_counts().sort_index()
    for cls, count in y_train_counts.items():
        pct = 100 * count / len(y_train)
        print(f"  Class {cls}: {count:3d} ({pct:5.1f}%)")

    print(f"\nClass Distribution (Test):")
    y_test_counts = pd.Series(y_test).value_counts().sort_index()
    for cls, count in y_test_counts.items():
        pct = 100 * count / len(y_test)
        print(f"  Class {cls}: {count:3d} ({pct:5.1f}%)")

    # Save statistics
    stats_file = output_dir / "preprocessing_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("FEATURE PREPROCESSING STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_test shape:  {X_test.shape}\n")
        f.write(f"y_train shape: {y_train.shape}\n")
        f.write(f"y_test shape:  {y_test.shape}\n\n")
        f.write("Train Class Distribution:\n")
        for cls, count in y_train_counts.items():
            pct = 100 * count / len(y_train)
            f.write(f"  Class {cls}: {count} ({pct:.1f}%)\n")
        f.write("\nTest Class Distribution:\n")
        for cls, count in y_test_counts.items():
            pct = 100 * count / len(y_test)
            f.write(f"  Class {cls}: {count} ({pct:.1f}%)\n")

    print(f"\n✓ Statistics saved to {stats_file}")


def run_all(output_dir: str | None = None) -> None:
    """Run all feature engineering steps."""
    run_feature_selection(output_dir)
    run_preprocessing(output_dir)
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline for heart disease dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["selection", "preprocessing", "all"],
        default="all",
        help="Which feature engineering step to run (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/features",
        help="Output directory for results (default: artifacts/features)"
    )

    args = parser.parse_args()

    try:
        if args.mode == "selection":
            run_feature_selection(args.output_dir)
        elif args.mode == "preprocessing":
            run_preprocessing(args.output_dir)
        else:  # all
            run_all(args.output_dir)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
