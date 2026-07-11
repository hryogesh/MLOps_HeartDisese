from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats


def tune_hyperparameters(df: pd.DataFrame) -> Dict[str, object]:
    """Tune logistic regression hyperparameters using a small randomized search."""
    feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal"
    ]
    X = df[feature_columns]
    y = df["target"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, feature_columns)])
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    param_dist = {
        "classifier__C": stats.loguniform(0.01, 10),
        "classifier__solver": ["liblinear", "lbfgs"],
    }
    randomized_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    randomized_search.fit(X, y)

    return {
        "search_type": "randomized_search",
        "best_params": {key.replace("classifier__", ""): value for key, value in randomized_search.best_params_.items()},
        "best_score": float(randomized_search.best_score_),
    }
