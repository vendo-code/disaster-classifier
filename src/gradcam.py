"""
gradcam.py
Implementasi Gradient-weighted Class Activation Mapping (Grad-CAM)
untuk visualisasi area yang memengaruhi prediksi model MobileNetV2.

Referensi: Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


# Layer konvolusi terakhir MobileNetV2 sebelum GlobalAveragePooling
LAST_CONV_LAYER = "out_relu"  # layer terakhir sebelum pooling di MobileNetV2


def get_gradcam_heatmap(
    model: tf.keras.Model,
    preprocessed_image: np.ndarray,
    predicted_class_idx: int,
    last_conv_layer_name: str = LAST_CONV_LAYER,
) -> np.ndarray:
    """
    Menghitung heatmap Grad-CAM.

    Args:
        model: model Keras MobileNetV2
        preprocessed_image: array (1, 224, 224, 3) sudah dinormalisasi
        predicted_class_idx: indeks kelas yang ingin divisualisasikan
        last_conv_layer_name: nama layer konvolusi terakhir

    Returns:
        heatmap: np.ndarray shape (H, W) nilai float [0, 1]
    """
    # Buat model grad: input → [activations, predictions]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(preprocessed_image, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        # Nilai output untuk kelas prediksi
        loss = predictions[:, predicted_class_idx]

    # Gradien output kelas terhadap feature map konvolusi
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling pada gradien → bobot per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Heatmap = kombinasi linear feature map dan bobotnya
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisasi ke [0, 1] dengan ReLU untuk hanya ambil kontribusi positif
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()

    # Hindari pembagian nol
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Menggabungkan heatmap Grad-CAM dengan gambar asli sebagai overlay.

    Args:
        original_image: PIL Image gambar asli (sebelum preprocessing)
        heatmap: np.ndarray shape (H, W) nilai [0, 1]
        alpha: opacity heatmap (0=transparan, 1=penuh) — default 0.4
        colormap: colormap OpenCV untuk visualisasi (default JET)

    Returns:
        PIL Image hasil overlay
    """
    # Resize heatmap ke ukuran gambar asli
    orig_w, orig_h = original_image.size
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (orig_w, orig_h))

    # Terapkan colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Konversi gambar asli ke array
    original_array = np.array(original_image.convert("RGB"))

    # Overlay: blend gambar asli dan heatmap
    overlay = cv2.addWeighted(original_array, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(overlay)


def generate_gradcam_result(
    model: tf.keras.Model,
    original_image: Image.Image,
    preprocessed_image: np.ndarray,
    predicted_class_idx: int,
    alpha: float = 0.4,
) -> tuple[Image.Image, np.ndarray]:
    """
    Fungsi utama: hitung Grad-CAM dan buat overlay.

    Args:
        model: model Keras
        original_image: PIL Image asli
        preprocessed_image: array (1, 224, 224, 3)
        predicted_class_idx: indeks kelas prediksi
        alpha: opacity overlay (0.0 - 1.0)

    Returns:
        (overlay_image, heatmap_raw)
    """
    heatmap = get_gradcam_heatmap(model, preprocessed_image, predicted_class_idx)
    overlay = overlay_heatmap_on_image(original_image, heatmap, alpha=alpha)
    return overlay, heatmap
