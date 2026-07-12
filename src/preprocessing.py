"""
preprocessing.py
Modul preprocessing citra sebelum inferensi model MobileNetV2.
"""

import numpy as np
from PIL import Image
import io


# Ukuran input model MobileNetV2
TARGET_SIZE = (224, 224)

# Label kelas sesuai urutan output model
CLASS_NAMES = [
    "Damaged_Infrastructure",
    "Fire_Disaster",
    "Human_Damage",
    "Land_Disaster",
    "Non_Damage",
    "Water_Disaster",
]

CLASS_LABELS_ID = {
    "Damaged_Infrastructure": "Infrastruktur Rusak",
    "Fire_Disaster": "Bencana Kebakaran",
    "Human_Damage": "Korban Manusia",
    "Land_Disaster": "Bencana Tanah",
    "Non_Damage": "Tidak Ada Kerusakan",
    "Water_Disaster": "Bencana Air",
}


def load_image_from_upload(uploaded_file) -> Image.Image:
    """
    Memuat gambar dari file upload Streamlit.

    Args:
        uploaded_file: objek file dari st.file_uploader

    Returns:
        PIL Image dalam mode RGB
    """
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocessing citra sesuai standar MobileNetV2:
    - Resize ke 224x224
    - Konversi ke array NumPy
    - Normalisasi menggunakan preprocess_input MobileNetV2 (skala [-1, 1])
    - Tambah dimensi batch

    Args:
        image: PIL Image dalam mode RGB

    Returns:
        np.ndarray shape (1, 224, 224, 3) siap untuk inferensi
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Resize
    image_resized = image.resize(TARGET_SIZE, Image.LANCZOS)

    # Ke array
    img_array = np.array(image_resized, dtype=np.float32)

    # Tambah dimensi batch
    img_array = np.expand_dims(img_array, axis=0)

    # Normalisasi MobileNetV2 → skala [-1, 1]
    img_array = preprocess_input(img_array)

    return img_array


def validate_image_file(uploaded_file) -> tuple[bool, str]:
    """
    Validasi file upload:
    - Harus berformat JPG/PNG
    - Ukuran maksimal 10MB

    Args:
        uploaded_file: objek file dari st.file_uploader

    Returns:
        (valid: bool, pesan_error: str)
    """
    MAX_SIZE_MB = 10
    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

    if uploaded_file.type not in ALLOWED_TYPES:
        return False, f"Format file tidak didukung: {uploaded_file.type}. Gunakan JPG atau PNG."

    # Cek ukuran (dalam bytes)
    uploaded_file.seek(0, 2)  # seek ke akhir
    file_size_bytes = uploaded_file.tell()
    uploaded_file.seek(0)  # reset ke awal

    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > MAX_SIZE_MB:
        return False, f"Ukuran file terlalu besar: {file_size_mb:.1f}MB. Maksimal 10MB."

    return True, ""
