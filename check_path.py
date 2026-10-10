from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "UrbanSound8K.csv"
DATA_PATH = PROJECT_ROOT / "data" / "audio"


def main() -> None:
    print("--- Dataset path check ---")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata CSV not found at {METADATA_PATH}.")

    metadata = pd.read_csv(METADATA_PATH)
    if metadata.empty:
        raise ValueError(f"Metadata CSV is empty: {METADATA_PATH}.")

    row = next(metadata.itertuples(index=False))
    expected_path = DATA_PATH / f"fold{row.fold}" / str(row.slice_file_name)
    print(f"Checking sample: {expected_path}")
    if expected_path.is_file():
        print("Sample audio file found.")
    else:
        raise FileNotFoundError(f"Sample audio file not found at {expected_path}.")


if __name__ == "__main__":
    main()