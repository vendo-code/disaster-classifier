"""
pages/2_Info_Kelas.py
Halaman yang menampilkan deskripsi dan ciri visual 6 kelas bencana.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="Info Kelas Bencana — BencanaLens",
    page_icon="🌋",
    layout="wide",
)

from assets.class_info.class_data import CLASS_INFO

st.markdown(
    """
    <style>
    .class-card {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: white;
    }
    .class-card h3 {
        font-size: 1.3rem;
        margin-bottom: 0.3rem;
    }
    .class-card .subtitle {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-bottom: 0.8rem;
    }
    .class-card .description {
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }
    .class-card .detail-box {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        margin-bottom: 0.5rem;
    }
    .class-card .warning-note {
        background: rgba(255, 193, 7, 0.25);
        border-left: 3px solid #ffc107;
        border-radius: 0 6px 6px 0;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌋 Info Kelas Bencana")
st.caption(
    "Penjelasan 6 kategori bencana yang dapat dikenali model, "
    "lengkap dengan ciri visual dan fokus area Grad-CAM."
)
st.markdown("---")

# Tampilkan 2 kartu per baris
class_keys = list(CLASS_INFO.keys())
pairs = [(class_keys[i], class_keys[i + 1] if i + 1 < len(class_keys) else None)
         for i in range(0, len(class_keys), 2)]

for key_left, key_right in pairs:
    col_left, col_right = st.columns(2, gap="medium")

    for col, key in [(col_left, key_left), (col_right, key_right)]:
        if key is None:
            continue
        info = CLASS_INFO[key]
        with col:
            has_note = "catatan_model" in info and info["catatan_model"]
            note_html = ""
            if has_note:
                note_html = f"""
                <div class="warning-note">
                    ⚠️ <strong>Catatan Model:</strong> {info['catatan_model']}
                </div>
                """

            st.markdown(
                f"""
                <div class="class-card" style="background: linear-gradient(135deg, {info['warna']}dd, {info['warna']}99);">
                    <h3>{info['emoji']} {info['label_id']}</h3>
                    <div class="subtitle">({key})</div>
                    <div class="description">{info['deskripsi']}</div>
                    <div class="detail-box">
                        🔥 <strong>Fokus Grad-CAM:</strong><br>
                        {info['ciri_gradcam']}
                    </div>
                    <div class="detail-box">
                        📌 <strong>Contoh Kejadian:</strong><br>
                        {info['contoh_kejadian']}
                    </div>
                    {note_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")

# Tabel ringkasan
st.markdown("### 📋 Ringkasan Kelas")

ringkasan = []
for key, info in CLASS_INFO.items():
    ringkasan.append({
        "Emoji": info["emoji"],
        "Nama Kelas": key,
        "Label (ID)": info["label_id"],
        "Contoh Kejadian": info["contoh_kejadian"],
    })

import pandas as pd
df_ringkasan = pd.DataFrame(ringkasan)
st.dataframe(df_ringkasan, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    """
    <div style="color: #6c757d; font-size: 0.85rem;">
    Deskripsi kelas disusun berdasarkan naskah jurnal dan karakteristik dataset CDD
    (Niloy et al., 2021). Kategori ini bersifat visual — model mengenali pola piksel,
    bukan konteks semantik mendalam.
    </div>
    """,
    unsafe_allow_html=True,
)
