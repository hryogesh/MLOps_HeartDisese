from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris


def load_dataset(path=None) -> pd.DataFrame:
    """Load the public Iris dataset and save it locally for reproducibility."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "iris.csv"

    if not Path(path).exists():
        iris = load_iris(as_frame=True)
        df = iris.frame
        df = df.rename(columns={"target": "label"})
        df["label"] = df["label"].astype(int)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return pd.read_csv(path)
