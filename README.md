# RoomGuard — UAS Workshop Machine Learning for IoT (Kelompok 12)

**Judul:** RoomGuard: Sistem Web Klasifikasi Keberadaan Orang dalam Ruangan
Menggunakan Decision Tree Berbasis Sensor Ambience Smartphone

**Kelompok 12:** Evodia Maribet Kristina Manalu & Hasbi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evodia24trk-ship-it/RoomGuard/blob/main/RoomGuard_Colab.ipynb)

**Mata kuliah:** Workshop Machine Learning for IoT — UAS

Sistem web yang mengklasifikasikan status keberadaan orang dalam ruangan ke
dalam 2 kelas (**Terisi / Kosong**) berdasarkan sinyal sensor intensitas
cahaya (Lux) dan tingkat suara (dB) pada smartphone. Model dikembangkan
menggunakan dataset publik UCI Room Occupancy Estimation Dataset (75%),
digabungkan dan diuji dengan rekaman sensor cahaya dan mikrofon smartphone
menggunakan aplikasi phyphox pada interval pencatatan 30 detik dengan
pelabelan manual (25% private). Hasil analisis ditampilkan melalui aplikasi web.

- Dataset publik: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation

## Hasil (terverifikasi, data uji n = 600)

| Metrik | Nilai |
|---|---|
| Akurasi | **94,83%** (569/600) |
| Macro F1-score | 93,56% |
| Macro precision / recall | 92,59% / 94,69% |
| CV macro-F1 (5-fold) | 95,24% |
| Confusion matrix | TN 418, FP 22, FN 9, TP 151 |
| Model terbaik | Decision Tree — gini, max_depth = 7, min_samples_leaf = 10 |

## Cara menjalankan

```bash
pip install -r requirements.txt
python src/build_dataset.py     # 1.500 publik + 500 private -> 2.000 baris
python src/train_model.py       # latih + evaluasi, simpan model/
python app/app.py               # buka http://127.0.0.1:5000
```

Coba: Light **120** Lux + Sound **56** dB → Occupied (100%).
Coba: Light **5** Lux + Sound **36** dB → Empty (100%).

## Isi folder

| Path | Isi |
|---|---|
| `app/` | Aplikasi web Flask (English UI): prediksi, batch CSV, dashboard, history CRUD (SQLite), decision path |
| `src/build_dataset.py` | Membangun dataset gabungan 75/25 dari UCI + phyphox |
| `src/train_model.py` | Grid search + 5-fold CV, evaluasi, simpan model & gambar |
| `dataset/` | `combined_dataset_2000.csv`, `public_uci_sample_1500.csv`, `private_phyphox_500.csv`, `LABELING_STANDARD.md`, template lembar kehadiran, `DATASET_README.md` (cara memverifikasi dataset) |
| `dataset/uci864_original_10129.csv` | File asli dari link UCI (10.129 baris, tanpa diubah). Setiap baris sampel publik bisa dilacak ke file ini lewat kolom `uci_row_id` — lihat `dataset/DATASET_README.md`. |
| `model/` | `decision_tree_model.pkl`, `metrics.json`, confusion matrix / tree / feature importance / flowchart / arsitektur |
| `docs/` | Laporan (25 hlm, .docx + .pdf), PPT (10 slide), Logbook (.xlsx), screenshot tampilan web |
| `dataset/uci864_original_10129.csv` | File asli UCI 864 (10.129 baris) sebagai sumber |

## Standar pelabelan (catatan dosen)

Label **tidak pernah** ditentukan dari nilai sensor. Aturannya ada di
`dataset/LABELING_STANDARD.md`: Occupied = minimal 1 orang di dalam ruangan
selama interval 30 detik penuh, dihitung observer dan ditulis di lembar
kehadiran **sebelum** nilai sensor dibaca. Interval transisi dibuang,
dua observer harus sepakat, label tidak pernah diubah karena nilai sensornya
"kelihatan salah".

## ⚠️ Yang wajib dilakukan sebelum presentasi

1. **Ganti data private simulasi dengan rekaman phyphox asli.**
   File `dataset/private_phyphox_500.csv` saat ini dibangkitkan mengikuti
   protokol dan distribusi data (seed tetap, dapat direproduksi) agar sistem
   bisa dilatih dan didemonstrasikan sekarang — ini **bukan** rekaman asli
   Evodia/Hasbi. Rekam dengan phyphox (Light + Sound Meter, 30 detik),
   isi lembar kehadiran (`dataset/private_collection_log_TEMPLATE.csv`),
   simpan sebagai `dataset/private_phyphox_REAL.csv` (kolom `timestamp,
   Light_Lux, Sound_dB, Occupied`), lalu:
   ```bash
   python src/build_dataset.py --real-private dataset/private_phyphox_REAL.csv
   python src/train_model.py
   ```
   Angka di laporan/PPT harus diperbarui dari `model/metrics.json` yang baru.
2. **Lengkapi nama lengkap & NIM Hasbi** di cover laporan dan PPT
   (saat ini hanya tertulis "Hasbi").
3. **Kumpulkan H-1 ke Classroom:** PPT, Laporan, link dataset (bagikan
   akses ke `warnia@pcr.ac.id` dan `dara@pcr.ac.id`), link script Python,
   dan Logbook — sesuai petunjuk UAS.
