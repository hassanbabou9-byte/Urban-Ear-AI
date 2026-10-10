from os import PathLike
from typing import BinaryIO

import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION_SECONDS = 3
MEL_BANDS = 128
SPECTROGRAM_WIDTH = 128


def load_audio_features(
    file_path: str | PathLike[str] | BinaryIO,
) -> tuple[np.ndarray, int, np.ndarray]:
    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION_SECONDS,
    )
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=MEL_BANDS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < SPECTROGRAM_WIDTH:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, SPECTROGRAM_WIDTH - mel_db.shape[1])),
            mode="constant",
        )
    else:
        mel_db = mel_db[:, :SPECTROGRAM_WIDTH]

    return audio, sample_rate, mel_db.astype(np.float32)


def prepare_model_input(mel_db: np.ndarray) -> np.ndarray:
    mel_db = np.asarray(mel_db, dtype=np.float32)
    if mel_db.ndim not in (2, 3) or mel_db.shape[-2:] != (
        MEL_BANDS,
        SPECTROGRAM_WIDTH,
    ):
        raise ValueError(
            "Expected a 128x128 spectrogram or a batch of 128x128 spectrograms."
        )

    return np.repeat(mel_db[..., np.newaxis], 3, axis=-1)
