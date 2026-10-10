from pathlib import Path

import numpy as np
from tensorflow.keras import layers, models, applications

from src.audio_features import prepare_model_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "processed_data.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "urban_ear_model.h5"
NUM_CLASSES = 10


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Preprocessed data not found at {DATA_PATH}. Run `python -m src.preprocess` first."
        )

    with np.load(DATA_PATH) as data:
        if "fold" not in data:
            raise ValueError(
                "Preprocessed data does not contain fold assignments. "
                "Regenerate it with `python -m src.preprocess`."
            )
        X = data["x"]
        y = data["y"].astype(np.int64)
        folds = data["fold"].astype(np.int64)

    train_mask = (folds >= 1) & (folds <= 8)
    validation_mask = folds == 9
    test_mask = folds == 10
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError(
            "Expected UrbanSound8K fold assignments 1–10, with folds 9 and 10 "
            "reserved for validation and testing."
        )
    if not np.all((y >= 0) & (y < NUM_CLASSES)):
        raise ValueError(f"Expected class IDs in the range 0–{NUM_CLASSES - 1}.")

    X_train = prepare_model_input(X[train_mask])
    X_validation = prepare_model_input(X[validation_mask])
    X_test = prepare_model_input(X[test_mask])
    y_train, y_validation, y_test = (
        y[train_mask],
        y[validation_mask],
        y[test_mask],
    )

    base_model = applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(
        f"Training on {len(y_train)} samples, validating on {len(y_validation)}, "
        f"testing on {len(y_test)}."
    )
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_validation, y_validation),
    )
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to {MODEL_PATH}.")


if __name__ == "__main__":
    main()