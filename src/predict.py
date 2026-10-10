import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.audio_features import load_audio_features, prepare_model_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "urban_ear_model.h5"
CLASSES = [
    "Air Conditioner",
    "Car Horn",
    "Children Playing",
    "Dog Bark",
    "Drilling",
    "Engine Idling",
    "Gun Shot",
    "Jackhammer",
    "Siren",
    "Street Music",
]


def run_prediction(file_path: str | Path) -> None:
    file_path = Path(file_path)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")

    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, mel_db = load_audio_features(file_path)
    processed_audio = prepare_model_input(mel_db)[np.newaxis, ...]
    prediction = model.predict(processed_audio, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    print("-" * 30)
    print(f"File: {file_path.name}")
    print(f"AI Prediction: {CLASSES[class_idx]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict path/to/your/audio.wav")
        sys.exit(2)
    else:
        run_prediction(sys.argv[1])