"""
inference.py
Modul untuk memuat model dan menjalankan prediksi klasifikasi bencana.
"""

import numpy as np
import os
import streamlit as st
from pathlib import Path


MODEL_PATH = Path(__file__).parent.parent / "models" / "disaster_mobilenetv2_best.h5"

# URL download otomatis jika model tidak ada di lokal
# (Isi dengan URL Hugging Face Hub atau Google Drive jika diperlukan)
MODEL_DOWNLOAD_URL = ""  # Isi URL jika model di-host eksternal


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Memuat model Keras dari file .h5.
    Menggunakan st.cache_resource agar tidak reload tiap interaksi.

    Returns:
        tf.keras.Model atau None jika gagal
    """
    import tensorflow as tf

    if not MODEL_PATH.exists():
        # Coba download otomatis jika URL tersedia
        if MODEL_DOWNLOAD_URL:
            _download_model(MODEL_DOWNLOAD_URL, MODEL_PATH)
        else:
            return None

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


def _download_model(url: str, save_path: Path):
    """
    Download model dari URL eksternal (Hugging Face Hub / Google Drive).
    """
    import requests

    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner("Mengunduh model... (hanya sekali, harap tunggu)"):
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        st.error(f"Gagal mengunduh model: {e}")


def predict(model, preprocessed_image: np.ndarray) -> tuple[np.ndarray, str, float]:
    """
    Menjalankan inferensi pada gambar yang sudah dipreproses.

    Args:
        model: model Keras yang sudah dimuat
        preprocessed_image: array shape (1, 224, 224, 3)

    Returns:
        (probabilities, predicted_class, confidence_score)
        - probabilities: np.ndarray shape (6,) nilai probabilitas tiap kelas
        - predicted_class: nama kelas dengan probabilitas tertinggi
        - confidence_score: nilai probabilitas tertinggi dalam persen (0-100)
    """
    from src.preprocessing import CLASS_NAMES

    # Inferensi
    predictions = model.predict(preprocessed_image, verbose=0)
    probabilities = predictions[0]  # shape (6,)

    # Kelas prediksi
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx]) * 100

    return probabilities, predicted_class, confidence
