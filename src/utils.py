"""
utils.py
Fungsi utilitas umum untuk aplikasi BencanaLens.
"""

import numpy as np
from PIL import Image
import base64
import io
from datetime import datetime


def image_to_base64(image: Image.Image) -> str:
    """
    Konversi PIL Image ke string base64 (untuk embedding di HTML/CSS).
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def format_confidence(confidence: float) -> str:
    """
    Format nilai confidence ke string persen dengan 2 desimal.
    """
    return f"{confidence:.2f}%"


def get_confidence_color(confidence: float) -> str:
    """
    Mengembalikan warna berdasarkan tingkat confidence:
    - Hijau  : ≥ 80%
    - Kuning : 50–79%
    - Merah  : < 50%
    """
    if confidence >= 80:
        return "#2ecc71"
    elif confidence >= 50:
        return "#f39c12"
    else:
        return "#e74c3c"


def get_risk_level(class_name: str, confidence: float) -> tuple[str, str]:
    """
    Menentukan level risiko bencana berdasarkan kelas dan confidence.

    Returns:
        (level_text, level_color)
    """
    # Kelas yang butuh perhatian tinggi
    HIGH_RISK_CLASSES = {
        "Fire_Disaster",
        "Water_Disaster",
        "Land_Disaster",
        "Damaged_Infrastructure",
        "Human_Damage",
    }

    if class_name == "Non_Damage":
        return "Aman", "#2ecc71"

    if class_name in HIGH_RISK_CLASSES:
        if confidence >= 80:
            return "Risiko Tinggi", "#e74c3c"
        elif confidence >= 50:
            return "Risiko Sedang", "#f39c12"
        else:
            return "Perlu Verifikasi", "#95a5a6"

    return "Tidak Diketahui", "#95a5a6"


def create_scan_record(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    probabilities: np.ndarray,
    overlay_image: Image.Image = None,
) -> dict:
    """
    Membuat record hasil scan untuk disimpan di session state.

    Returns:
        dict berisi semua info hasil scan
    """
    from src.preprocessing import CLASS_NAMES, CLASS_LABELS_ID

    timestamp = datetime.now().strftime("%H:%M:%S")
    label_id = CLASS_LABELS_ID.get(predicted_class, predicted_class)
    risk_level, risk_color = get_risk_level(predicted_class, confidence)

    return {
        "timestamp": timestamp,
        "image": image,
        "overlay_image": overlay_image,
        "predicted_class": predicted_class,
        "label_id": label_id,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i]) * 100
            for i in range(len(CLASS_NAMES))
        },
        "risk_level": risk_level,
        "risk_color": risk_color,
    }


def thumbnail(image: Image.Image, size: tuple = (80, 80)) -> Image.Image:
    """
    Membuat thumbnail dari gambar untuk ditampilkan di riwayat scan.
    """
    img_copy = image.copy()
    img_copy.thumbnail(size, Image.LANCZOS)
    return img_copy
