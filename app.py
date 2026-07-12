"""
app.py — BencanaLens
Aplikasi demo klasifikasi citra bencana menggunakan MobileNetV2 + Grad-CAM.

Cara menjalankan:
    streamlit run app.py

Referensi paper:
    "Klasifikasi Citra Bencana Alam Menggunakan Arsitektur MobileNetV2
    melalui Transfer Learning Dua Fase dan Visualisasi Grad-CAM"
    (Syarif, Mahendra, Ramadhani, Yudha — Universitas Dian Nuswantoro)
"""

import sys
import os

# Pastikan folder src bisa diimport
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd

from src.preprocessing import (
    load_image_from_upload,
    preprocess_image,
    validate_image_file,
    CLASS_NAMES,
    CLASS_LABELS_ID,
)
from src.inference import load_model, predict
from src.gradcam import generate_gradcam_result
from src.utils import (
    format_confidence,
    get_confidence_color,
    get_risk_level,
    create_scan_record,
    thumbnail,
)

# ──────────────────────────────────────────────
# Konfigurasi halaman
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BencanaLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS kustom
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 1rem;
        color: #a0aec0;
        margin: 0;
    }

    /* Kartu hasil */
    .result-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Badge kelas */
    .class-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
    }

    /* Progress bar confidence */
    .conf-bar-container {
        background: #e9ecef;
        border-radius: 6px;
        height: 10px;
        margin-top: 4px;
    }
    .conf-bar-fill {
        height: 10px;
        border-radius: 6px;
    }

    /* Sidebar */
    .sidebar-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }

    /* Riwayat item */
    .history-item {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.6rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background 0.2s;
    }
    .history-item:hover {
        background: #f1f3f5;
    }

    /* Pesan error */
    .error-box {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        padding: 1rem;
        color: #c53030;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

if "selected_history_idx" not in st.session_state:
    st.session_state.selected_history_idx = None

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 BencanaLens</h1>
        <p>Klasifikasi Citra Bencana Alam · MobileNetV2 + Transfer Learning + Grad-CAM</p>
        <p style="font-size:0.8rem; margin-top:0.5rem; color:#718096;">
            Demo aplikasi pendamping jurnal ilmiah · Universitas Dian Nuswantoro
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar — Riwayat Scan
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🕒 Riwayat Scan")
    st.caption("Hasil scan sesi ini (tidak disimpan setelah browser ditutup)")

    if not st.session_state.scan_history:
        st.info("Belum ada riwayat scan.")
    else:
        for i, record in enumerate(reversed(st.session_state.scan_history)):
            idx = len(st.session_state.scan_history) - 1 - i
            is_selected = st.session_state.selected_history_idx == idx

            col_img, col_info = st.columns([1, 2])
            with col_img:
                thumb = thumbnail(record["image"], (60, 60))
                st.image(thumb, use_container_width=True)
            with col_info:
                st.markdown(
                    f"**{record['label_id']}**  \n"
                    f"🕐 {record['timestamp']}  \n"
                    f"Confidence: {record['confidence']:.1f}%"
                )
                if st.button("Lihat", key=f"hist_{idx}"):
                    st.session_state.selected_history_idx = idx
                    st.session_state.current_result = record
                    st.rerun()

            st.divider()

        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state.scan_history = []
            st.session_state.selected_history_idx = None
            st.session_state.current_result = None
            st.rerun()

    st.markdown("---")
    st.markdown("**Navigasi**")
    st.page_link("app.py", label="🏠 Beranda / Klasifikasi", icon="🔍")
    st.page_link("pages/1_Tentang_Model.py", label="📊 Tentang Model", icon="📊")
    st.page_link("pages/2_Info_Kelas.py", label="🌋 Info Kelas Bencana", icon="🌋")

# ──────────────────────────────────────────────
# Muat model
# ──────────────────────────────────────────────
with st.spinner("Memuat model klasifikasi..."):
    model = load_model()

if model is None:
    st.error(
        "⚠️ **Model tidak ditemukan!**\n\n"
        "Letakkan file `disaster_mobilenetv2_best.h5` di folder `models/`.\n\n"
        "Jika model di-host secara eksternal, isi `MODEL_DOWNLOAD_URL` di `src/inference.py`."
    )
    st.stop()

# ──────────────────────────────────────────────
# Layout utama — Upload & Hasil
# ──────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

# ── Kolom kiri: upload gambar ──
with col_upload:
    st.subheader("📤 Upload Gambar")
    st.caption("Format: JPG / PNG · Ukuran maks: 10 MB")

    uploaded_file = st.file_uploader(
        label="Pilih atau seret gambar bencana di sini",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Validasi file
        valid, err_msg = validate_image_file(uploaded_file)
        if not valid:
            st.markdown(
                f'<div class="error-box">⚠️ {err_msg}</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        # Muat dan tampilkan preview
        original_image = load_image_from_upload(uploaded_file)
        st.image(original_image, caption="Gambar asli", use_container_width=True)

        # Tombol klasifikasi
        classify_btn = st.button(
            "🔍 Klasifikasikan",
            type="primary",
            use_container_width=True,
        )

        if classify_btn:
            with st.spinner("Menganalisis gambar..."):
                # Preprocessing
                preprocessed = preprocess_image(original_image)

                # Inferensi
                probabilities, predicted_class, confidence = predict(model, preprocessed)
                predicted_idx = CLASS_NAMES.index(predicted_class)

                # Grad-CAM (alpha default 0.4, bisa diubah oleh user)
                alpha_val = 0.4
                overlay_img, heatmap = generate_gradcam_result(
                    model,
                    original_image,
                    preprocessed,
                    predicted_idx,
                    alpha=alpha_val,
                )

            # Simpan ke session state
            record = create_scan_record(
                image=original_image,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                overlay_image=overlay_img,
            )
            st.session_state.scan_history.append(record)
            st.session_state.current_result = record
            st.rerun()

    else:
        # Placeholder
        st.markdown(
            """
            <div style="
                border: 2px dashed #ced4da;
                border-radius: 12px;
                padding: 3rem 2rem;
                text-align: center;
                color: #6c757d;
                background: #f8f9fa;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🌋</div>
                <div style="font-size: 1rem; font-weight: 500;">Upload foto bencana untuk memulai</div>
                <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                    Mendukung: JPG, JPEG, PNG
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Kolom kanan: hasil klasifikasi ──
with col_result:
    st.subheader("📊 Hasil Klasifikasi")

    result = st.session_state.current_result

    if result is None:
        st.info("Upload gambar dan klik **Klasifikasikan** untuk melihat hasil.")
    else:
        # ── Bagian atas: kelas prediksi & confidence ──
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        label_id = result["label_id"]
        risk_level, risk_color = result["risk_level"], result["risk_color"]
        conf_color = get_confidence_color(confidence)

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 0.3rem;">Kelas Prediksi</div>
                <div style="font-size: 1.6rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {label_id}
                </div>
                <div style="font-size: 0.8rem; color: #6c757d; margin-bottom: 1rem;">
                    ({predicted_class})
                </div>
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
                    <span style="
                        background: {conf_color};
                        color: white;
                        padding: 0.25rem 0.75rem;
                        border-radius: 20px;
                        font-size: 0.9rem;
                        font-weight: 600;
                    ">
                        Confidence: {confidence:.2f}%
                    </span>
                    <span style="
                        background: {risk_color};
                        color: white;
                        padding: 0.25rem 0.75rem;
                        border-radius: 20px;
                        font-size: 0.9rem;
                        font-weight: 600;
                    ">
                        {risk_level}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Distribusi probabilitas ──
        st.markdown("**Distribusi Probabilitas Semua Kelas**")
        probs = result["probabilities"]

        prob_data = []
        for cls_name in CLASS_NAMES:
            prob_data.append({
                "Kelas": CLASS_LABELS_ID.get(cls_name, cls_name),
                "Probabilitas (%)": round(probs[cls_name], 2),
            })
        df_probs = pd.DataFrame(prob_data).sort_values("Probabilitas (%)", ascending=False)

        st.bar_chart(
            df_probs.set_index("Kelas")["Probabilitas (%)"],
            use_container_width=True,
            height=200,
            color="#3498db",
        )

        # ── Grad-CAM visualisasi ──
        st.markdown("**Visualisasi Grad-CAM**")
        st.caption("Area yang disorot menunjukkan bagian gambar yang paling memengaruhi prediksi.")

        alpha_slider = st.slider(
            "Opacity heatmap",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            key="alpha_slider",
        )

        # Toggle foto asli vs overlay
        show_overlay = st.toggle("Tampilkan Grad-CAM overlay", value=True)

        if show_overlay and result.get("overlay_image") is not None:
            # Jika slider berubah, re-generate overlay
            if abs(alpha_slider - 0.4) > 0.01:
                from src.preprocessing import preprocess_image as _preprocess
                from src.gradcam import generate_gradcam_result as _gradcam
                new_overlay, _ = _gradcam(
                    model,
                    result["image"],
                    _preprocess(result["image"]),
                    CLASS_NAMES.index(result["predicted_class"]),
                    alpha=alpha_slider,
                )
                st.image(new_overlay, caption="Grad-CAM Heatmap Overlay", use_container_width=True)
            else:
                st.image(
                    result["overlay_image"],
                    caption="Grad-CAM Heatmap Overlay",
                    use_container_width=True,
                )
        else:
            st.image(result["image"], caption="Foto Asli", use_container_width=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        BencanaLens · Demo aplikasi pendamping jurnal ilmiah<br>
        Model: MobileNetV2 · Dataset: CDD (Niloy et al., 2021) · Akurasi: 94,31%<br>
        Universitas Dian Nuswantoro — 2024
    </div>
    """,
    unsafe_allow_html=True,
)
