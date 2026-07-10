import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_data import load_dataset
from src.data.validate import validate_dataset
from src.features.feature_selection import select_features
from src.features.preprocess import prepare_features
from src.models.hyperparameter_tuning import tune_hyperparameters
from src.models.predict import predict
from src.models.train_model import ModelTrainer
from src.monitoring.model_registry import register_model


def test_dataset_loads():
    df = load_dataset(path="data/heart_disease.csv")
    assert not df.empty
    assert df.shape[0] > 200
    assert "target" in df.columns


def test_validation_reports_dataset_health():
    df = load_dataset(path="data/heart_disease.csv")
    report = validate_dataset(df)
    assert report["row_count"] == df.shape[0]
    assert "missing_values" in report


def test_preprocessing_returns_expected_shapes():
    df = load_dataset(path="data/heart_disease.csv")
    prepared = prepare_features(df)
    assert prepared["X_train"].shape[1] >= 8
    assert prepared["X_test"].shape[1] >= 8
    assert prepared["y_train"].shape[0] == prepared["X_train"].shape[0]


def test_prediction_returns_expected_keys():
    result = predict([63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0])
    assert set(result.keys()) == {"prediction", "probability"}


def test_model_training_writes_model_artifact(tmp_path):
    model_path = tmp_path / "model.joblib"
    trainer = ModelTrainer(model_path=str(model_path))
    payload = trainer.train()
    assert model_path.exists()
    assert payload["metrics"]["accuracy"] > 0.7


def test_feature_selection_reduces_features():
    df = load_dataset(path="data/heart_disease.csv")
    selection = select_features(df)
    assert len(selection["selected_columns"]) < df.shape[1] - 1


def test_hyperparameter_tuning_returns_best_params():
    df = load_dataset(path="data/heart_disease.csv")
    tuning = tune_hyperparameters(df)
    assert "C" in tuning["best_params"]


def test_model_registration_returns_run_id(tmp_path):
    model_path = tmp_path / "registry_model.joblib"
    trainer = ModelTrainer(model_path=str(model_path))
    payload = trainer.train()
    run_id = register_model(payload, artifact_path=str(tmp_path))
    assert run_id
