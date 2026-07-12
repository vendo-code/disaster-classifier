"""
pages/1_Tentang_Model.py
Halaman yang menampilkan detail arsitektur, metrik evaluasi, dan keterbatasan model.
Ditujukan untuk dosen penguji dan reviewer jurnal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Tentang Model — BencanaLens",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .section-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Tentang Model")
st.caption(
    "Halaman ini menyajikan detail teknis dan metrik evaluasi model untuk keperluan "
    "verifikasi oleh dosen penguji dan reviewer jurnal."
)

# ──────────────────────────────────────────────
# Metrik ringkasan (Tabel 7 paper)
# ──────────────────────────────────────────────
st.markdown("## 🎯 Metrik Evaluasi Keseluruhan")
st.caption("Hasil evaluasi pada test set — Fase 2 fine-tuning (Tabel 7 paper)")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        '<div class="metric-card"><h2>94,31%</h2><p>Test Accuracy</p></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card"><h2>94,25%</h2><p>F1-Score (weighted)</p></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        '<div class="metric-card"><h2>0,9924</h2><p>AUC-ROC (macro)</p></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        '<div class="metric-card"><h2>94,98%</h2><p>Val. Accuracy (Fase 2)</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ──────────────────────────────────────────────
# Metrik per kelas (Tabel 8 paper)
# ──────────────────────────────────────────────
st.markdown("## 📋 Metrik Per Kelas (Tabel 8 Paper)")

data_per_kelas = {
    "Kelas": [
        "Damaged_Infrastructure",
        "Fire_Disaster",
        "Human_Damage",
        "Land_Disaster",
        "Non_Damage",
        "Water_Disaster",
        "**Weighted Avg**",
    ],
    "Precision": [0.9254, 0.9689, 0.9412, 0.8241, 0.9711, 0.9563, 0.9430],
    "Recall": [0.9127, 0.9745, 0.9318, 0.7892, 0.9789, 0.9612, 0.9431],
    "F1-Score": [0.9190, 0.9717, 0.9365, 0.6705, 0.9750, 0.9587, 0.9425],
    "Support": [812, 743, 681, 598, 827, 716, 4377],
}

df_kelas = pd.DataFrame(data_per_kelas)

# Highlight baris Land_Disaster karena F1 terendah
def highlight_row(row):
    if "Land_Disaster" in str(row["Kelas"]):
        return ["background-color: #fff3cd"] * len(row)
    elif "Weighted" in str(row["Kelas"]):
        return ["background-color: #e8f5e9; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_kelas.style.apply(highlight_row, axis=1).format(
        {
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "Support": "{:,.0f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    '<div class="warning-box">⚠️ <strong>Catatan:</strong> '
    'Land_Disaster memiliki F1-Score terendah (0,6705). Lihat bagian "Keterbatasan Model" di bawah '
    'untuk penjelasan lebih lengkap.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ──────────────────────────────────────────────
# Arsitektur & Training
# ──────────────────────────────────────────────
st.markdown("## 🏗️ Arsitektur & Skema Training")

col_arch, col_train = st.columns(2, gap="large")

with col_arch:
    st.markdown("### Arsitektur Model")
    st.markdown(
        """
        <div class="section-box">
        <strong>Base Model:</strong> MobileNetV2 (pretrained ImageNet)<br><br>
        <strong>Input:</strong> 224 × 224 × 3 (RGB)<br>
        <strong>Normalisasi:</strong> preprocess_input MobileNetV2 → skala [-1, 1]<br><br>
        <strong>Head klasifikasi:</strong><br>
        &nbsp;&nbsp;• GlobalAveragePooling2D<br>
        &nbsp;&nbsp;• Dense 256, ReLU + Dropout 0.5<br>
        &nbsp;&nbsp;• Dense 128, ReLU + Dropout 0.3<br>
        &nbsp;&nbsp;• Dense 6, Softmax (output 6 kelas)<br><br>
        <strong>Layer Grad-CAM:</strong> <code>out_relu</code> (konvolusi terakhir MobileNetV2)
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_train:
    st.markdown("### Skema Transfer Learning Dua Fase")
    st.markdown(
        """
        <div class="section-box">
        <strong>Fase 1 — Feature Extraction:</strong><br>
        &nbsp;&nbsp;• Base MobileNetV2 <em>dibekukan</em> (frozen)<br>
        &nbsp;&nbsp;• Hanya head klasifikasi yang dilatih<br>
        &nbsp;&nbsp;• Optimizer: Adam (lr=1e-3)<br>
        &nbsp;&nbsp;• Epoch: 30 + early stopping<br><br>
        <strong>Fase 2 — Fine-tuning:</strong><br>
        &nbsp;&nbsp;• 50 layer terakhir base MobileNetV2 <em>dibuka</em> (unfrozen)<br>
        &nbsp;&nbsp;• Optimizer: Adam (lr=1e-5, learning rate kecil)<br>
        &nbsp;&nbsp;• Epoch: 30 + early stopping<br>
        &nbsp;&nbsp;• Val. Accuracy terbaik: <strong>94,98%</strong><br><br>
        <strong>Bobot disimpan:</strong> <code>disaster_mobilenetv2_best.h5</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
st.markdown("## 📂 Dataset")
st.markdown(
    """
    <div class="section-box">
    <strong>Nama Dataset:</strong> CDD — Comprehensive Disaster Dataset<br>
    <strong>Sumber:</strong> Kaggle (Niloy et al., 2021)<br>
    <strong>Total Citra:</strong> 13.557 gambar<br>
    <strong>Jumlah Kelas:</strong> 6 kelas bencana<br><br>
    <strong>Pembagian Data:</strong>
    <table style="margin-top:0.5rem; border-collapse: collapse; width: 100%;">
      <tr style="background:#e9ecef;">
        <th style="padding:0.3rem 0.6rem; text-align:left;">Split</th>
        <th style="padding:0.3rem 0.6rem; text-align:left;">Proporsi</th>
        <th style="padding:0.3rem 0.6rem; text-align:left;">Jumlah</th>
      </tr>
      <tr>
        <td style="padding:0.3rem 0.6rem;">Training</td>
        <td style="padding:0.3rem 0.6rem;">70%</td>
        <td style="padding:0.3rem 0.6rem;">~9.490</td>
      </tr>
      <tr style="background:#f8f9fa;">
        <td style="padding:0.3rem 0.6rem;">Validation</td>
        <td style="padding:0.3rem 0.6rem;">15%</td>
        <td style="padding:0.3rem 0.6rem;">~2.034</td>
      </tr>
      <tr>
        <td style="padding:0.3rem 0.6rem;">Test</td>
        <td style="padding:0.3rem 0.6rem;">15%</td>
        <td style="padding:0.3rem 0.6rem;">~2.034</td>
      </tr>
    </table><br>
    <em>Sumber dataset wajib dicantumkan sesuai etika publikasi: Niloy et al. (2021),
    "CDD: A Large Scale Comprehensive Disaster Dataset", Kaggle.</em>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────
st.markdown("## 🔥 Implementasi Grad-CAM")
st.markdown(
    """
    <div class="section-box">
    Grad-CAM (Gradient-weighted Class Activation Mapping) digunakan untuk menghasilkan
    <em>heatmap</em> yang menunjukkan area mana dalam gambar yang paling memengaruhi keputusan prediksi model.<br><br>
    <strong>Layer target:</strong> <code>out_relu</code> — layer konvolusi terakhir MobileNetV2
    sebelum GlobalAveragePooling<br>
    <strong>Proses:</strong><br>
    &nbsp;&nbsp;1. Hitung gradien output kelas target terhadap feature map konvolusi<br>
    &nbsp;&nbsp;2. Global average pooling pada gradien → bobot tiap channel<br>
    &nbsp;&nbsp;3. Kombinasi linear feature map × bobot, lalu ReLU<br>
    &nbsp;&nbsp;4. Resize heatmap ke ukuran gambar asli<br>
    &nbsp;&nbsp;5. Overlay dengan colormap JET<br><br>
    <strong>Referensi:</strong> Selvaraju, R.R. et al. (2017). "Grad-CAM: Visual Explanations
    from Deep Networks via Gradient-based Localization." ICCV.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ──────────────────────────────────────────────
# Keterbatasan Model (transparansi untuk reviewer)
# ──────────────────────────────────────────────
st.markdown("## ⚠️ Keterbatasan Model")
st.markdown(
    """
    <div class="warning-box">
    <strong>1. Confusion Land_Disaster ↔ Damaged_Infrastructure</strong><br>
    Kedua kelas ini memiliki ciri visual yang tumpang tindih — keduanya menampilkan lingkungan
    yang hancur atau tidak teratur. Model sering ragu antara dua kelas ini, yang tercermin
    pada F1-Score Land_Disaster yang lebih rendah (0,6705) dibanding kelas lain.<br><br>
    <strong>2. Ketergantungan pada kualitas gambar</strong><br>
    Model dilatih pada dataset dengan kondisi gambar tertentu. Gambar dengan resolusi sangat
    rendah, blur berat, atau sudut pandang tidak umum dapat menurunkan akurasi.<br><br>
    <strong>3. Bukan sistem produksi</strong><br>
    Aplikasi ini adalah <em>demo</em> untuk keperluan akademik. Tidak direkomendasikan
    untuk digunakan sebagai satu-satunya alat identifikasi bencana di lapangan tanpa
    validasi lebih lanjut.<br><br>
    <strong>4. Out-of-distribution</strong><br>
    Model mungkin tidak generalizes dengan baik pada gambar bencana dari wilayah dengan
    karakteristik visual yang sangat berbeda dari dataset CDD.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ──────────────────────────────────────────────
# Perbandingan Fase 1 vs Fase 2 (Tabel 9 paper)
# ──────────────────────────────────────────────
st.markdown("## 📈 Perbandingan Fase Training")

data_fase = {
    "Fase": ["Fase 1 (Feature Extraction)", "Fase 2 (Fine-tuning)"],
    "Val. Accuracy": ["91,23%", "94,98%"],
    "Test Accuracy": ["90,87%", "94,31%"],
    "F1-Score": ["90,72%", "94,25%"],
    "AUC-ROC": ["0,9811", "0,9924"],
}

df_fase = pd.DataFrame(data_fase)
st.dataframe(df_fase, use_container_width=True, hide_index=True)

st.caption(
    "Fine-tuning Fase 2 memberikan peningkatan signifikan di semua metrik, "
    "mengkonfirmasi efektivitas transfer learning dua fase untuk klasifikasi citra bencana."
)

# ──────────────────────────────────────────────
# Footer referensi
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="color: #6c757d; font-size: 0.85rem;">
    <strong>Referensi paper:</strong><br>
    Syarif, Mahendra, Ramadhani, Yudha (2024).
    "Klasifikasi Citra Bencana Alam Menggunakan Arsitektur MobileNetV2
    melalui Transfer Learning Dua Fase dan Visualisasi Grad-CAM."
    Universitas Dian Nuswantoro.<br><br>
    <strong>Dataset:</strong> Niloy et al. (2021). CDD: Comprehensive Disaster Dataset. Kaggle.
    </div>
    """,
    unsafe_allow_html=True,
)
