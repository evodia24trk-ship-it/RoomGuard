# Logbook Pengerjaan Proyek UAS — RoomGuard

**Kelompok 12:** Evodia, Hasbi
**Mata Kuliah:** Workshop Machine Learning for IoT
**Periode:** November 2026
**Judul:** RoomGuard: Sistem Web Klasifikasi Keberadaan Orang dalam Ruangan Menggunakan Decision Tree Berbasis Sensor Ambience Smartphone

Logbook ini mencatat enam pertemuan pengerjaan proyek, dari perumusan ide sampai laporan selesai. Tanggal dan kegiatan ditulis sesuai urutan pengerjaan yang sebenarnya.

---

## Pertemuan 1 — Senin, 3 November 2026: Perumusan Ide dan Pembagian Tugas

**Kegiatan:**
- Diskusi awal tentang topik UAS: disepakati untuk mengambil tema klasifikasi keberadaan orang di ruangan berbasis sensor ponsel, karena topiknya dekat dengan kehidupan sehari-hari dan datanya bisa dikumpulkan sendiri.
- Menetapkan judul, rumusan masalah, tujuan, dan batasan masalah: klasifikasi biner (Terisi/Kosong), dua fitur (cahaya dan suara), model Decision Tree.
- Mencari dataset publik dan menemukan UCI Room Occupancy Estimation Dataset #864.
- Pembagian tugas: Evodia menangani data dan model, Hasbi menangani aplikasi web; laporan dan presentasi dikerjakan bersama.

**Hasil:**
- Judul dan kerangka Bab I selesai dirumuskan.
- Link dataset publik dicatat: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation

**Kendala:**
- Sempat ragu apakah dua fitur saja cukup untuk klasifikasi yang baik. Diputuskan untuk tetap mencoba, dengan catatan hasilnya akan dievaluasi secara jujur — kalau memang kurang, keterbatasan itu akan ditulis di laporan.

---

## Pertemuan 2 — Sabtu, 8 November 2026: Pengambilan Data Privat dengan Phyphox

**Kegiatan:**
- Mengunduh file mentah dataset UCI (10.129 baris) dan memeriksa isinya: kolom tanggal, count, empat kolom cahaya, dan empat kolom suara.
- Mempelajari aplikasi Phyphox: cara merekam intensitas cahaya dan tingkat suara, serta cara mengekspor hasil rekaman ke CSV.
- Menyusun lembar observasi sederhana (waktu, ada/tidak ada orang di ruangan) untuk pelabelan manual.
- Melakukan 2 sesi pertama pengambilan data privat: ponsel diletakkan di atas meja, interval pencatatan 30 detik, satu sesi saat ruangan terisi dan satu sesi saat ruangan kosong.

**Hasil:**
- File mentah UCI berhasil diunduh dan dibaca; komposisinya 8.228 baris Kosong dan 1.901 baris Terisi.
- Prosedur perekaman Phyphox sudah berjalan; 2 sesi pertama menghasilkan data yang bisa diekspor.

**Kendala:**
- Pada percobaan pertama, sensor cahaya sempat tertutup sebagian oleh posisi ponsel. Posisi ponsel diperbaiki (layar menghadap ke atas, tidak tertutup benda lain) dan sesi diulang.

---

## Pertemuan 3 — Kamis, 13 November 2026: Menyelesaikan Data Privat dan Preprocessing

**Kegiatan:**
- Melanjutkan pengambilan data privat sampai total 5 sesi dan 500 baris (250 Terisi, 250 Kosong).
- Mencocokkan setiap baris rekaman Phyphox dengan lembar observasi berdasarkan waktu, lalu memberi label Terisi/Kosong. Label diambil dari observasi, bukan dari nilai sensor.
- Menghitung fitur dari data publik: Light_Lux = max(S1–S4_Light), Sound_dB = 20×log10(max(S1–S4_Sound)) + 60; label publik dari count > 0.
- Menggabungkan kedua sumber menjadi 2.000 baris (1.500 publik + 500 privat), memeriksa nilai kosong dan nilai di luar rentang (Lux 0–280, dB 25–85).

**Hasil:**
- Dataset gabungan 2.000 baris selesai dan bersih, dengan komposisi 1.000 Terisi dan 1.000 Kosong.
- Pembagian stratified seed 42 selesai: train 1.400, validation 300, test 300, masing-masing seimbang.

**Kendala:**
- Beberapa baris data privat awalnya sulit dicocokkan dengan lembar observasi karena jam ponsel dan jam pencatat berbeda beberapa detik. Diputuskan untuk mencocokkan berdasarkan urutan waktu dan selisih yang paling dekat, dan selisih ini dicatat sebagai catatan keterbatasan.

---

## Pertemuan 4 — Rabu, 18 November 2026: Pelatihan dan Evaluasi Model

**Kegiatan:**
- Menulis notebook Google Colab: pemuatan data, pembagian data, GridSearchCV 5-fold untuk Decision Tree, pelatihan model terbaik, dan evaluasi.
- Menjalankan GridSearchCV dan mencatat hasilnya: kombinasi terbaik gini, max_depth 5, min_samples_leaf 10, dengan CV accuracy 97,29%.
- Mengevaluasi model pada data uji (n = 300) dan menghitung metrik serta confusion matrix.
- Menganalisis feature importance dan aturan utama pohon, serta memecah kinerja data uji berdasarkan sumber (publik vs privat).

**Hasil:**
- Test: accuracy 95,33%, precision 95,95%, recall 94,67%, F1 95,30%; train 97,86%, validation 95,00%.
- Confusion matrix [[144, 6], [8, 142]]; per kelas Kosong P/R/F1 0,95/0,96/0,95 dan Terisi 0,96/0,95/0,95.
- Per sumber: publik 96,94% (n = 229), privat 90,14% (n = 71); Kosong privat terendah 87,10%.
- Feature importance: Sound_dB 0,9006, Light_Lux 0,0994; aturan utama: ≤ 41,5 dB cenderung Kosong, > 46,35 dB cenderung Terisi.

**Kendala:**
- Akurasi pada data privat ternyata lebih rendah dari data publik, terutama kelas Kosong. Setelah diperiksa, penyebabnya kemungkinan suara dari luar ruangan yang ikut terekam ponsel. Hasil ini tidak ditutup-tutupi; justru dijadikan bahan analisis kesalahan dan keterbatasan di Bab IV.

---

## Pertemuan 5 — Minggu, 22 November 2026: Pembangunan Aplikasi Web

**Kegiatan:**
- Membangun aplikasi web RoomGuard dengan Flask dan SQLite: halaman Dashboard, Predict, History, Dataset, Model, dan About (frontend berbahasa Inggris).
- Menghubungkan model Decision Tree yang sudah dilatih ke endpoint `/api/predict`, lengkap dengan rekomendasi hemat energi berdasarkan hasil prediksi.
- Mengimplementasikan penyimpanan riwayat prediksi ke SQLite beserta fitur hapus satu entri dan hapus semua.
- Menguji seluruh halaman dan tombol secara manual dengan daftar periksa: prediksi, penyimpanan riwayat, penghapusan, dan input di luar rentang wajar.

**Hasil:**
- Aplikasi web berjalan secara lokal tanpa error; semua menu dan fitur berfungsi, termasuk rekomendasi hemat energi yang sesuai dengan hasil prediksi.
- Model yang dipakai web sama dengan model yang dievaluasi di notebook, sehingga angka yang ditampilkan di halaman Model konsisten dengan laporan.

**Kendala:**
- Sempat ada ketidaksesuaian format antara input form dan yang diharapkan endpoint API (angka dibaca sebagai teks). Diperbaiki dengan konversi tipe data di backend sebelum dikirim ke model.

---

## Pertemuan 6 — Selasa, 24 November 2026: Penyusunan Laporan, PPT, dan Finalisasi

**Kegiatan:**
- Menyusun laporan lengkap Bab I–V dan Daftar Pustaka berdasarkan angka-angka yang sudah didapatkan pada Pertemuan 4; seluruh angka diperiksa ulang agar konsisten antara laporan, notebook, dan halaman Model di web.
- Menyusun outline presentasi 12 slide dan membagi peran presenter.
- Mengunggah kode, notebook, dan dataset ke repositori GitHub proyek.
- Pemeriksaan akhir: ejaan dan konsistensi angka, tautan dataset UCI, dan kelengkapan lampiran.

**Hasil:**
- Laporan, outline PPT, dan logbook selesai.
- Repositori GitHub berisi kode aplikasi web, notebook Google Colab, dan dataset gabungan.

**Kendala:**
- Beberapa angka di draf awal laporan tidak konsisten (misalnya pembulatan presisi yang berbeda antara Bab IV dan Bab V). Semua angka diselaraskan ke nilai yang dipakai di laporan ini sebelum finalisasi.

---

*Logbook ini disusun berdasarkan catatan pengerjaan kelompok selama November 2026 dan dapat diverifikasi dengan riwayat commit pada repositori GitHub proyek.*
