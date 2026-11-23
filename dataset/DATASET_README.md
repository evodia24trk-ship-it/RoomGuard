# Dataset — RoomGuard (Kelompok 12)

## Link yang diberikan ke dosen

**https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation**
(UCI Room Occupancy Estimation Dataset, ID 864, diakses 23 November 2026, lisensi CC BY 4.0)

## File di folder ini

| File | Isi |
|---|---|
| `uci864_original_10129.csv` | **File asli dari link UCI di atas, tanpa diubah satu byte pun.** 10.129 baris, 19 kolom: `Date, Time, S1_Temp…S4_Temp, S1_Light…S4_Light, S1_Sound…S4_Sound, S5_CO2, S5_CO2_Slope, S6_PIR, S7_PIR, Room_Occupancy_Count` (0–3 orang). Buka file ini untuk membandingkan dengan dataset UCI. |
| `public_uci_sample_1500.csv` | Sampel 1.500 baris dari file asli (stratified, random_state=42). Setiap baris membawa `uci_row_id` = nomor baris asli (0-based) di file asli, dan `Room_Occupancy_Count` aslinya. |
| `public_uci_sample_1500_full.csv` | Sampel yang sama dengan **semua kolom asli** + kolom turunan, supaya tidak perlu menebak-nebak. |
| `private_phyphox_500.csv` | 500 baris private (phyphox, interval 30 detik). Lihat catatan kejujuran di bawah. |
| `combined_dataset_2000.csv` | Gabungan 1.500 + 500 = 2.000 baris yang dipakai melatih model. Kolom `uci_row_id` / `Room_Occupancy_Count` kosong untuk baris private. |
| `LABELING_STANDARD.md` | Standar pelabelan (label dari lembar kehadiran, bukan dari nilai sensor). |
| `private_collection_log_TEMPLATE.csv` | Template lembar kehadiran untuk rekaman phyphox asli. |

## Cara memverifikasi (untuk dosen / siapa pun)

1. Buka `uci864_original_10129.csv`, cari baris `uci_row_id` — misalnya `uci_row_id = 0`
   adalah baris pertama file asli (`2017/12/22 10:49:41, …, Room_Occupancy_Count = 1`).
2. Di `public_uci_sample_1500.csv`, cari baris dengan `uci_row_id` yang sama.
   Periksa: `Light_Lux` = rata-rata `S1_Light, S2_Light, S3_Light, S4_Light`
   pada baris asli itu; `Occupied` = 1 jika `Room_Occupancy_Count > 0`, selain itu 0.
3. `Sound_dB` = `20·log10(mean(S1_Sound…S4_Sound)/0,05) + 35` — suara UCI
   tersimpan dalam Volt (bukan dB), jadi dikonversi agar satuannya sama dengan
   phyphox. Rumus ini ada di `src/build_dataset.py` (fungsi `sound_v_to_db`).
4. Jumlah per kelas sampel publik: Empty 1.218, Occupied 282 — sama dengan
   proporsi file asli (Empty 8.228 / Occupied 1.901 dari 10.129).
5. Jalankan `python src/build_dataset.py` — skrip ini memeriksa ulang
   (`traceability check OK`) bahwa semua 1.500 baris publik cocok dengan
   file asli berdasarkan `uci_row_id`.

## Catatan kejujuran data private

`private_phyphox_500.csv` saat ini dibangkitkan mengikuti protokol phyphox
(seed tetap, dapat direproduksi dengan `src/build_dataset.py`) agar sistem
dapat dilatih dan didemonstrasikan — ini **bukan** rekaman phyphox asli.
Sebelum presentasi, ganti dengan rekaman asli (lihat `README.md` di root
proyek) dan latih ulang.
