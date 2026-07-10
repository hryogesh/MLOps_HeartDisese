from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


DATASET_ARCHIVE = Path("../Downloads/heart+disease.zip")


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the Cleveland heart disease dataset from the attached archive or a local CSV."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "heart_disease.csv"
    else:
        path = Path(path)

    if not path.exists():
        if not DATASET_ARCHIVE.exists():
            raise FileNotFoundError(
                "The attached heart disease archive was not found. Expected it at ../Downloads/heart+disease.zip"
            )

        with zipfile.ZipFile(DATASET_ARCHIVE) as archive:
            raw_lines = archive.read("processed.cleveland.data").decode("utf-8", "ignore").splitlines()

        columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        df = pd.DataFrame([line.split(",") for line in raw_lines if line.strip()], columns=columns)
        df = df.apply(pd.to_numeric, errors="coerce")
        df["target"] = df["target"].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return pd.read_csv(path)
