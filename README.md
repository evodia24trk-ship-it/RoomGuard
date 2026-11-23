# RoomGuard — UAS Workshop Machine Learning for IoT (Kelompok 12)

**Judul:** RoomGuard: Sistem Web Klasifikasi Keberadaan Orang dalam Ruangan
Menggunakan Decision Tree Berbasis Sensor Ambience Smartphone

**Kelompok 12:** Evodia Maribet Kristina Manalu & Hasbi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evodia24trk-ship-it/RoomGuard/blob/main/RoomGuard_Colab.ipynb)

**Mata kuliah:** Workshop Machine Learning for IoT — UAS

## Coba langsung (tanpa install apa-apa)

- **Web demo:** https://muse.ai/s/roomguard-qxd6xtx0qxvc36xq/vHaVcXNJtWeQ3qwUuwXzrGtNIjjfz-R2e8xBzuYJTek
- **Web demo (GitHub Pages):** aktifkan dulu di Settings → Pages → Deploy from a branch → `main` → folder `/docs`, lalu buka `https://evodia24trk-ship-it.github.io/RoomGuard/` — file-nya `docs/index.html`, isinya pohon Decision Tree yang sama persis (diekspor dari model terlatih), jadi prediksinya sama dengan aplikasi Flask-nya.
- **Notebook Colab:** [RoomGuard_Colab.ipynb](https://colab.research.google.com/github/evodia24trk-ship-it/RoomGuard/blob/main/RoomGuard_Colab.ipynb) — jalankan dari atas ke bawah, Runtime → Run all.

## Ceritanya singkat

Kami sering melihat lampu dan AC menyala di ruangan yang sudah kosong. Kamera
terasa terlalu invasif dan sensor PIR cuma tahu ada gerakan, bukan ada orang.
Jadi kami coba cara yang lebih sederhana: HP yang ditaruh di meja, sensor
cahaya (Lux) dan mikrofonnya (dB) dibaca lewat phyphox tiap 30 detik, lalu
model Decision Tree menebak ruangan itu Terisi atau Kosong. Hasilnya tampil
di aplikasi web.

Datasetnya 2.000 baris: 1.500 publik dari UCI Room Occupancy Estimation
(https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) dan
500 private dari phyphox. Labelnya tidak pernah ditentukan dari nilai sensor —
aturannya ada di `dataset/LABELING_STANDARD.md` (lihat bawah).

## Hasil (data uji n = 600)

| Metrik | Nilai |
|---|---|
| Akurasi | **94,83%** (569/600) |
| Macro F1-score | 93,56% |
| Macro precision / recall | 92,59% / 94,69% |
| CV macro-F1 (5-fold) | 95,24% |
| Confusion matrix | TN 418, FP 22, FN 9, TP 151 |
| Model terbaik | Decision Tree — gini, max_depth = 7, min_samples_leaf = 10 |

Yang paling mengejutkan kami: suara ternyata jauh lebih menentukan daripada
cahaya (feature importance Sound 86,24%, Light 13,76%). Masuk akal kalau
dipikir-pikir — lampu bisa menyala di ruangan kosong, tapi ruangan berisik
hampir selalu ada orangnya. Justru di situlah 31 baris yang salah
diklasifikasi: kebanyakan lampu menyala saat ruangan kosong.

## Cara menjalankan di laptop

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
| `docs/index.html` | Versi web statis (untuk GitHub Pages) — pohon yang sama, jalan di browser tanpa server |
| `src/build_dataset.py` | Membangun dataset gabungan 75/25 dari UCI + phyphox |
| `src/train_model.py` | Grid search + 5-fold CV, evaluasi, simpan model & gambar |
| `dataset/` | `combined_dataset_2000.csv`, `public_uci_sample_1500.csv`, `private_phyphox_500.csv`, `LABELING_STANDARD.md`, template lembar kehadiran, `DATASET_README.md` (cara memverifikasi dataset) |
| `dataset/uci864_original_10129.csv` | File asli dari link UCI (10.129 baris, tanpa diubah). Setiap baris sampel publik bisa dilacak ke file ini lewat kolom `uci_row_id` — lihat `dataset/DATASET_README.md`. |
| `model/` | `decision_tree_model.pkl`, `metrics.json`, confusion matrix / tree / feature importance / flowchart / arsitektur |
| `docs/` | Laporan (25 hlm, .docx + .pdf), PPT (10 slide), Logbook (.xlsx), screenshot tampilan web |
| `RoomGuard_Colab.ipynb` | Notebook Colab: download UCI → bangun fitur → latih → evaluasi → coba prediksi |

## Cara dosen memeriksa datasetnya

File asli UCI (10.129 baris) ada di repo tanpa diubah:
`dataset/uci864_original_10129.csv`. Tiap baris di
`dataset/public_uci_sample_1500.csv` punya `uci_row_id` (nomor baris di file
asli) dan `Room_Occupancy_Count` aslinya. Ambil satu baris, cocokkan ke file
asli, hitung rata-rata Light S1–S4 — harus sama dengan `Light_Lux` di sampel.
Langkah lengkapnya di `dataset/DATASET_README.md`.

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
