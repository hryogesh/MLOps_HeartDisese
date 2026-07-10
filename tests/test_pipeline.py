import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_dataset
from src.features.preprocess import prepare_features
from src.models.predict import predict


def test_dataset_loads():
    df = load_dataset(path="data/iris.csv")
    assert not df.empty
    assert "label" in df.columns


def test_preprocessing_returns_expected_shapes():
    df = load_dataset(path="data/iris.csv")
    prepared = prepare_features(df)
    assert prepared["X_train"].shape[1] == 4
    assert prepared["X_test"].shape[1] == 4
    assert prepared["y_train"].shape[0] == prepared["X_train"].shape[0]


def test_prediction_returns_expected_keys():
    result = predict([5.1, 3.5, 1.4, 0.2])
    assert set(result.keys()) == {"prediction", "probability"}
