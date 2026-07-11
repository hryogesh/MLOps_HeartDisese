from pathlib import Path

from src.data.load_data import load_dataset


def main() -> None:
    archive_path = Path("../Downloads/heart+disease.zip")
    if not archive_path.exists():
        raise FileNotFoundError("Please place the heart disease archive at ../Downloads/heart+disease.zip")

    output_path = Path("data/heart_disease.csv")
    load_dataset(path=str(output_path))
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
