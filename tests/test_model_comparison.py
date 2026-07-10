import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_dataset
from src.models.model_comparison import compare_models


def test_compare_models_returns_results():
    df = load_dataset(path="data/heart_disease.csv")
    comparison = compare_models(df)
    assert "results" in comparison
    assert "best_model" in comparison
    assert comparison["best_model"] in {"logistic_regression", "random_forest"}
