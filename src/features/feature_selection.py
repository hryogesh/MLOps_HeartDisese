from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def select_features(df: pd.DataFrame, k: int = 8) -> Dict[str, object]:
    """Select the most informative numerical features for classification."""
    feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal"
    ]
    X = df[feature_columns]
    y = df["target"]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
    ])
    pipeline.fit(X, y)
    selector = pipeline.named_steps["selector"]
    selected_columns = [col for col, selected in zip(feature_columns, selector.get_support()) if selected]

    return {
        "selected_columns": selected_columns,
        "selector": selector,
        "pipeline": pipeline,
    }
