# Folder `models/`

Letakkan file model Keras di sini:

```
models/
└── disaster_mobilenetv2_best.h5   ← bobot terbaik hasil fine-tuning Fase 2
```

## Cara mendapatkan file model

### Opsi 1 — Dari hasil training lokal
Jalankan notebook `notebooks/training_pipeline.ipynb`. 
File `disaster_mobilenetv2_best.h5` akan disimpan otomatis oleh `ModelCheckpoint`.

### Opsi 2 — Download dari Hugging Face Hub
Jika file terlalu besar untuk GitHub (>100MB), upload ke Hugging Face Hub dan isi
`MODEL_DOWNLOAD_URL` di `src/inference.py`:

```python
MODEL_DOWNLOAD_URL = "https://huggingface.co/<username>/<repo>/resolve/main/disaster_mobilenetv2_best.h5"
```

Model akan didownload otomatis saat pertama kali aplikasi dijalankan.

### Opsi 3 — Google Drive
Gunakan `gdown` untuk download dari Google Drive, lalu tambahkan logika download
di `src/inference.py`.

## Spesifikasi model

| Parameter | Nilai |
|---|---|
| Arsitektur | MobileNetV2 + custom head |
| Input size | 224 × 224 × 3 |
| Jumlah kelas | 6 |
| Format | Keras HDF5 (.h5) |
| Val. Accuracy | 94,98% |
| Test Accuracy | 94,31% |
