"""
Complete Data Pipeline Script
Runs the full data ingestion, validation, EDA, and preprocessing pipeline.

Usage:
    python scripts/data_pipeline.py              # Run entire pipeline
    python scripts/data_pipeline.py --step eda   # Run only EDA
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Execute a shell command and report status."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd="/Users/meghana_p/mlops-end-to-end-project")
    if result.returncode != 0:
        print(f"\n✗ {description} failed with exit code {result.returncode}")
        return result.returncode

    print(f"\n✓ {description} completed successfully")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete data pipeline for heart disease project"
    )
    parser.add_argument(
        "--step",
        choices=["data", "eda", "features", "all"],
        default="all",
        help="Which pipeline step to run (default: all)"
    )

    args = parser.parse_args()

    steps = []

    if args.step in ["data", "all"]:
        steps.append(
            ([sys.executable, "scripts/download_dataset.py"],
             "STEP 1: Download & Validate Dataset")
        )

    if args.step in ["eda", "all"]:
        steps.append(
            ([sys.executable, "scripts/eda_report.py"],
             "STEP 2: Exploratory Data Analysis (EDA)")
        )

    if args.step in ["features", "all"]:
        steps.append(
            ([sys.executable, "scripts/feature_engineering.py", "--mode", "all"],
             "STEP 3: Feature Engineering (Selection & Preprocessing)")
        )

    if not steps:
        print("No steps selected")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("DATA PIPELINE")
    print(f"{'=' * 70}")
    print(f"Selected steps: {args.step.upper()}")
    print(f"Total steps: {len(steps)}\n")

    exit_code = 0
    for i, (cmd, desc) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {desc}")
        if run_command(cmd, desc) != 0:
            exit_code = 1
            print(f"\n⚠ Pipeline stopped at step {i}")
            break

    if exit_code == 0:
        print(f"\n{'=' * 70}")
        print("DATA PIPELINE COMPLETE ✓")
        print(f"{'=' * 70}")
        print("\nNext steps:")
        print("  1. Review EDA reports in: artifacts/eda/")
        print("  2. Review feature selection in: artifacts/features/")
        print("  3. Run model training: python -m src.train")
        print(f"{'=' * 70}\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
