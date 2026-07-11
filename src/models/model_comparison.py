from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def compare_models(df: pd.DataFrame) -> Dict[str, object]:
    """Train and compare logistic regression and random forest classifiers with preprocessing and CV."""
    feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal"
    ]
    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, feature_columns)])

    models = {
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float((test_predictions == y_test).mean()),
            "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
        }

    best_model_name = max(results, key=lambda key: results[key]["accuracy"])
    return {
        "results": results,
        "best_model": best_model_name,
        "feature_columns": feature_columns,
        "tuning_note": "Models were trained with median imputation, standard scaling, and 5-fold stratified cross-validation.",
    }
