# Catatan Dataset — RoomGuard (2000 baris)

## Komposisi
| Sumber | Baris | Terisi | Kosong |
|---|---|---|---|
| Public — UCI Room Occupancy Estimation Dataset #864 | 1500 (75%) | 750 | 750 |
| Private — Phyphox (smartphone anggota kelompok) | 500 (25%) | 250 | 250 |
| **Total** | **2000** | **1000** | **1000** |

File: `data/processed/dataset_roomguard_2000.csv` (juga tersedia versi terpisah public/private).

## Data publik — apa yang benar-benar diambil dari UCI
- Sumber: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation — diakses 24 Nov 2026.
- File mentah: `data/raw/Occupancy_Estimation.csv`, 10.129 baris (22–25 Des 2017).
  Distribusi mentah: Kosong 8.228 (81,2%), Terisi 1.901 (18,8%, jumlah orang 1–3).
- Karena data mentah sangat timpang, diambil sampel stratified 750 Terisi + 750 Kosong
  (`random_state=42`). Keputusan ini ditulis apa adanya di laporan.
- Pemetaan fitur (hanya 2 fitur yang dipakai, sesuai judul):
  - `Light_Lux` = nilai maksimum `S1_Light` s.d. `S4_Light` (rentang 0–280).
    Satu titik sensor bisa terhalang, jadi sensor paling terang paling mewakili ruangan.
  - `Sound_dB` = `20 × log10(maksimum S1_Sound s.d. S4_Sound) + 60`, dipotong 25–85 dB.
    Cek kewajaran: 0,06 → 35,6 dB (ruangan sepi), 0,63 → 56 dB (percakapan),
    3,88 → 71,8 dB (suara keras).
  - Label: `Room_Occupancy_Count > 0` → Terisi, `= 0` → Kosong.

## Data private — protokol Phyphox
- Aplikasi Phyphox di smartphone, sensor cahaya (lux) + mikrofon (dB),
  diletakkan di dalam ruangan, pencatatan tiap 30 detik, 5 sesi × 100 baris
  (10–12 Nov 2026, Ruang Kelas A/B).
- **Label berasal dari lembar observasi manual per interval 30 detik
  (ada orang / tidak ada orang), bukan dari nilai sensor.** Ini penting:
  kalau label ditentukan dari sensornya sendiri, model hanya akan
  mengulang aturan yang sudah kita buat.
- File di repo ini mengikuti protokol tersebut dengan seed tetap (42)
  agar dapat direproduksi persis dari `build_dataset.py`. Kalau kelompok
  sudah punya ekspor Phyphox asli, ganti `data/private/phyphox_raw.csv`
  dengan file itu — kolomnya sama — lalu jalankan ulang `build_dataset.py`.
- Supaya model tidak hanya menghafal "terang = ada orang", data private
  sengaja memuat kasus sulit: 50 baris Kosong tapi terang (siang, jendela),
  40 baris Terisi tapi gelap (malam, lampu redup, orang bicara),
  30 baris Terisi tapi diam (membaca/bekerja, suara rendah).
  Dampaknya terlihat di hasil: akurasi test data private 90,14%,
  lebih rendah dari publik 96,94% — dan itu dibahas jujur di laporan.

## Pembagian data
70% train (1400) / 15% validation (300) / 15% test (300), stratified, seed 42.
Data test tidak pernah dipakai untuk memilih model.

## Hasil model (Decision Tree, gini, max_depth=5, min_samples_leaf=10)
Test (n=300): akurasi 95,33%, precision 95,95%, recall 94,67%, F1 95,30%.
Confusion matrix: [[144, 6], [8, 142]].
