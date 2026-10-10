from io import BytesIO
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from src.audio_features import load_audio_features, prepare_model_input
from src.predict import CLASSES

st.set_page_config(page_title="Urban Ear AI", layout="centered")
st.title("Urban Ear: AI Sound Classifier")
st.write("Upload a city sound clip and let the AI identify it.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "urban_ear_model.h5"


@st.cache_resource
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.is_file():
    st.error(f"Model not found at {MODEL_PATH}.")
    st.stop()

model = load_model()

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue(), format="audio/wav")
    try:
        audio, sample_rate, mel_db = load_audio_features(
            BytesIO(uploaded_file.getvalue())
        )
        input_data = prepare_model_input(mel_db)[np.newaxis, ...]
        prediction = model.predict(input_data, verbose=0)
    except Exception as exc:
        st.error(f"Could not process this audio file: {exc}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.write("Waveform")
        fig, ax = plt.subplots(figsize=(5, 3))
        librosa.display.waveshow(audio, sr=sample_rate, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.write("Mel-spectrogram")
        fig, ax = plt.subplots(figsize=(5, 3))
        librosa.display.specshow(mel_db, sr=sample_rate, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    class_idx = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][class_idx])
    st.success(
        f"Prediction: **{CLASSES[class_idx]}** ({confidence * 100:.2f}% confidence)"
    )