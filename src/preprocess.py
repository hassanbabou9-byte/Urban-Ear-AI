import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.audio_features import load_audio_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_PATH = PROJECT_ROOT / "data" / "audio"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "UrbanSound8K.csv"
OUTPUT_PATH = PROJECT_ROOT / "processed_data.npz"


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH)
    features: list[np.ndarray] = []
    labels: list[int] = []
    folds: list[int] = []
    failures: list[tuple[Path, str]] = []

    print("Starting preprocessing...")
    for row in tqdm(metadata.itertuples(index=False), total=len(metadata)):
        file_path = AUDIO_PATH / f"fold{row.fold}" / str(row.slice_file_name)
        try:
            _, _, mel_db = load_audio_features(file_path)
        except Exception as exc:
            failures.append((file_path, str(exc)))
            continue

        features.append(mel_db)
        labels.append(int(row.classID))
        folds.append(int(row.fold))

    if not features:
        raise RuntimeError(
            f"No audio samples were processed. Check the dataset under {AUDIO_PATH}."
        )

    np.savez_compressed(
        OUTPUT_PATH,
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        fold=np.asarray(folds, dtype=np.int64),
    )
    print(f"Processed {len(features)} of {len(metadata)} samples.")
    print(f"Saved features and fold assignments to {OUTPUT_PATH}.")

    if failures:
        print(f"Skipped {len(failures)} unreadable audio files:")
        for file_path, error in failures[:10]:
            print(f"  {file_path}: {error}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more.")


if __name__ == "__main__":
    main()