# 🔍 BencanaLens

**Aplikasi demo klasifikasi citra bencana alam berbasis deep learning.**

Demo pendamping jurnal:
> *"Klasifikasi Citra Bencana Alam Menggunakan Arsitektur MobileNetV2 melalui Transfer Learning Dua Fase dan Visualisasi Grad-CAM"*  
> Syarif, Mahendra, Ramadhani, Yudha — Universitas Dian Nuswantoro (2024)

---

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| Upload & Preview | Upload JPG/PNG, preview sebelum diproses |
| Klasifikasi | Prediksi 6 kelas bencana + confidence score |
| Distribusi Probabilitas | Bar chart probabilitas semua kelas |
| Grad-CAM | Heatmap overlay untuk explainability |
| Tentang Model | Metrik evaluasi lengkap untuk reviewer jurnal |
| Info Kelas | Deskripsi 6 kategori bencana |
| Riwayat Scan | Hasil scan dalam sesi berjalan |

## 🏷️ Kelas Bencana

1. 🏚️ **Damaged_Infrastructure** — Infrastruktur Rusak
2. 🔥 **Fire_Disaster** — Bencana Kebakaran
3. 🚑 **Human_Damage** — Korban Manusia
4. ⛰️ **Land_Disaster** — Bencana Tanah
5. ✅ **Non_Damage** — Tidak Ada Kerusakan
6. 🌊 **Water_Disaster** — Bencana Air

---

## 🚀 Cara Menjalankan

### 1. Clone repo

```bash
git clone https://github.com/<username>/bencanalens.git
cd bencanalens/disaster-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9–3.11 direkomendasikan (kompatibel dengan TensorFlow 2.15)

### 3. Letakkan file model

Salin file `disaster_mobilenetv2_best.h5` ke folder `models/`:

```
disaster-classifier/
└── models/
    └── disaster_mobilenetv2_best.h5  ← letakkan di sini
```

Lihat `models/README.md` untuk opsi download jika file di-host di Hugging Face Hub.

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📁 Struktur Folder

```
disaster-classifier/
├── app.py                          # Halaman utama (upload & klasifikasi)
├── requirements.txt
├── README.md
├── models/
│   └── disaster_mobilenetv2_best.h5
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Preprocessing citra
│   ├── inference.py                # Load model & prediksi
│   ├── gradcam.py                  # Komputasi Grad-CAM
│   └── utils.py                   # Utilitas umum
├── pages/
│   ├── 1_Tentang_Model.py          # Metrik evaluasi & arsitektur
│   └── 2_Info_Kelas.py            # Deskripsi 6 kelas bencana
├── assets/
│   ├── sample_images/              # Gambar contoh untuk demo
│   └── class_info/
│       └── class_data.py          # Data deskripsi kelas
├── notebooks/
│   └── training_pipeline.ipynb    # Notebook training (untuk reproduksi)
└── data/
    └── README.md
```

---

## 🛠️ Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/)
- **Model:** TensorFlow/Keras — MobileNetV2
- **Explainability:** Grad-CAM (custom implementation)
- **Image processing:** OpenCV, Pillow
- **Visualisasi:** Matplotlib, Streamlit native charts

---

## 📊 Performa Model

| Metrik | Nilai |
|---|---|
| Test Accuracy | **94,31%** |
| F1-Score (weighted) | **94,25%** |
| AUC-ROC (macro) | **0,9924** |
| Val. Accuracy (Fase 2) | **94,98%** |

> ⚠️ Land_Disaster memiliki F1-Score terendah (0,6705) karena overlap visual
> dengan Damaged_Infrastructure. Lihat halaman "Tentang Model" di aplikasi.

---

## ☁️ Deploy ke Streamlit Community Cloud

1. Push repo ke GitHub (pastikan `models/` diabaikan `.gitignore` jika file terlalu besar)
2. Isi `MODEL_DOWNLOAD_URL` di `src/inference.py` dengan URL Hugging Face Hub
3. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan repo, set `app.py` sebagai entry point
4. Deploy!

---

## 📖 Referensi

- Sandler, M. et al. (2018). MobileNetV2. *CVPR*.
- Selvaraju, R.R. et al. (2017). Grad-CAM. *ICCV*.
- Niloy et al. (2021). CDD Dataset. *Kaggle*.

---

*Dibuat untuk keperluan tugas UAS & lampiran jurnal ilmiah. Bukan sistem produksi.*
