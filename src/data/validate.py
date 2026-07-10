from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a simple validation report for the dataset."""
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "columns": df.columns.tolist(),
    }
