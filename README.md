# RoomGuard — Sistem Web Klasifikasi Keberadaan Orang dalam Ruangan

Decision Tree berbasis sensor ambience smartphone (cahaya / Lux dan suara / dB).
Kelompok 12: Evodia, Hasbi — Workshop Machine Learning for IoT, UAS, November 2026.

## Hasil singkat
- Dataset: **2000 baris** — 1500 Public UCI Room Occupancy Estimation Dataset #864 (75%)
  + 500 Private Phyphox, interval 30 detik (25%). Seimbang: 1000 Terisi / 1000 Kosong.
- Split: 70% train (1400) / 15% validation (300) / 15% test (300), stratified, seed 42.
- Model: Decision Tree (`gini`, `max_depth=5`, `min_samples_leaf=10`, GridSearchCV 5-fold, CV 97,29%).
- Test (n=300): **akurasi 95,33%, precision 95,95%, recall 94,67%, F1 95,30%**,
  confusion matrix [[144, 6], [8, 142]].

## Isi repo
| Folder / file | Isi |
|---|---|
| `app/` | Aplikasi web Flask (frontend Bahasa Inggris, prediksi + CRUD history SQLite) |
| `data/processed/dataset_roomguard_2000.csv` | Dataset gabungan 2000 baris |
| `data/raw/` | Dataset mentah UCI 864 |
| `notebooks/RoomGuard_UAS_Colab.ipynb` | Notebook Google Colab (jalankan di Colab, sudah dieksekusi & 0 error) |
| `models/` | Model `.pkl`, `metrics.json`, grafik confusion matrix / pohon / scatter |
| `docs/` | Laporan (Bab I–V), outline PPT, logbook |
| `build_dataset.py`, `train.py` | Script reproduksi dataset & pelatihan |

## Menjalankan web (lokal)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
python app/app.py
# buka http://127.0.0.1:5000
```

## Menjalankan ulang pelatihan
```bash
python build_dataset.py
python train.py
```

## Sumber data
- Publik: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation (diakses 24 Nov 2026)
- Private: rekaman Phyphox anggota kelompok — lihat `DATASET_NOTES.md` untuk protokol & pemetaan fitur.
