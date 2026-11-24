# Outline Presentasi UAS — RoomGuard

**Judul:** RoomGuard: Sistem Web Klasifikasi Keberadaan Orang dalam Ruangan Menggunakan Decision Tree Berbasis Sensor Ambience Smartphone
**Kelompok 12:** Evodia, Hasbi — Workshop Machine Learning for IoT, November 2026
**Durasi yang disarankan:** 12–15 menit (sekitar 1 menit per slide) + tanya jawab

---

## Slide 1 — Judul

- Judul lengkap proyek, nama anggota kelompok (Evodia, Hasbi), mata kuliah Workshop Machine Learning for IoT, November 2026.
- Satu kalimat pembuka: "Bisakah ponsel biasa menebak apakah sebuah ruangan sedang terisi atau kosong?"
- Poin bicara: sapa penguji, perkenalkan diri singkat, sampaikan bahwa presentasi ini akan menunjukkan jawabannya lewat data dan demo web.

## Slide 2 — Latar Belakang & Masalah

- Lampu dan pendingin ruangan sering menyala di ruangan kosong; sensor khusus mahal dan tidak ada di semua ruangan.
- Ponsel sudah punya sensor cahaya dan mikrofon — apakah dua sinyal itu saja cukup?
- Rumusan masalah: (1) klasifikasi Terisi/Kosong dari Lux dan dB, (2) seberapa akurat modelnya, (3) bagaimana menyajikannya lewat web.
- Poin bicara: tekankan motivasi hemat energi, bukan sekadar soal teknis.

## Slide 3 — Tujuan & Batasan

- Tujuan: bangun model Decision Tree, gabungkan data publik + privat, sajikan lewat aplikasi web.
- Batasan: hanya 2 kelas, hanya 2 fitur (Light_Lux, Sound_dB), hanya Decision Tree, data privat dari satu ruangan, web berjalan lokal.
- Poin bicara: batasan disampaikan di awal supaya penguji tahu ruang lingkupnya dan tidak mengharapkan yang di luar itu.

## Slide 4 — Dataset: Publik + Privat

- Total 2.000 baris: 1.500 publik UCI Room Occupancy Estimation Dataset #864 (75%) + 500 privat Phyphox (25%).
- File mentah UCI: 10.129 baris (8.228 Kosong, 1.901 Terisi).
- Data privat: 5 sesi, interval 30 detik, 250 Terisi / 250 Kosong, pelabelan dari lembar observasi — bukan dari nilai sensor.
- Poin bicara: jelaskan kenapa digabung — data publik memberi jumlah, data privat menguji kondisi nyata.

## Slide 5 — Fitur & Preprocessing

- Hanya 2 fitur: Light_Lux = max(S1–S4_Light), 0–280 Lux; Sound_dB = 20×log10(max(S1–S4_Sound)) + 60, 25–85 dB.
- Label publik: count > 0 → Terisi. Label privat: dari observasi.
- Tidak ada normalisasi — Decision Tree tidak butuh itu.
- Poin bicara: tunjukkan contoh satu baris data sebelum dan sesudah preprocessing.

## Slide 6 — Metodologi & Pembagian Data

- Alur: pengumpulan → pembersihan → fitur → split → latih (GridSearchCV 5-fold) → evaluasi → web.
- Split 70/15/15 stratified, seed 42: train 1.400, validation 300, test 300 — masing-masing seimbang.
- Poin bicara: jelaskan kenapa stratified — supaya tiap bagian punya proporsi kelas yang sama.

## Slide 7 — Model: Decision Tree

- Best params dari GridSearchCV: gini, max_depth 5, min_samples_leaf 10; CV accuracy 97,29%.
- Kenapa pohon tidak dibuat lebih dalam: akurasi latih naik, tapi validasi tidak ikut naik.
- [Sertakan screenshot: models/decision_tree.png — Visualisasi Decision Tree]
- Poin bicara: tunjukkan gambar pohonnya dan bacakan satu jalur keputusan dari akar sampai daun.

## Slide 8 — Hasil Utama (Test, n = 300)

- **Accuracy 95,33% | Precision 95,95% | Recall 94,67% | F1 95,30%**
- Train 97,86%, validation 95,00% — selisih kecil, tidak overfitting.
- Confusion matrix: [[144, 6], [8, 142]] — 286 benar, 14 salah.
- [Sertakan screenshot: models/confusion_matrix.png — Confusion matrix data uji; dan docs/screenshots/colab-results.png — Hasil eksekusi notebook Colab]
- Poin bicara: jangan hanya membacakan angka; jelaskan artinya — dari 300 ruangan, 286 ditebak dengan benar.

## Slide 9 — Analisis Kesalahan & Per Sumber Data

- 6 false positive (kosong dikira terisi — suara dari luar ruangan) dan 8 false negative (terisi tapi hening — orang membaca tanpa suara).
- Per sumber: publik 96,94% (n = 229), privat 90,14% (n = 71); Kosong privat terendah 87,10%.
- Feature importance: Sound_dB 0,9006, Light_Lux 0,0994. Aturan utama: ≤ 41,5 dB cenderung Kosong, > 46,35 dB cenderung Terisi, di antaranya cahaya membantu.
- Poin bicara: akui dengan jujur bahwa data privat lebih sulit — ini justru menunjukkan pengujian yang jujur.

## Slide 10 — Demo Aplikasi Web

- Flask + SQLite, frontend Bahasa Inggris, menu Dashboard / Predict / History / Dataset / Model / About.
- Demo langsung: masukkan Lux dan dB, tampilkan hasil + rekomendasi hemat energi, lalu cek History.
- [Sertakan screenshot: docs/screenshots/web-predict.png — Tampilan halaman prediksi web RoomGuard]
- Poin bicara: siapkan 2 contoh input sebelum presentasi — satu yang jelas Terisi dan satu yang jelas Kosong — supaya demo tidak gagal di depan penguji.

## Slide 11 — Keterbatasan

- Data privat dari satu ruangan dan satu ponsel; suara luar tidak bisa dibedakan, sehingga Kosong privat paling sering salah dikenali.
- Hanya 2 fitur; input web masih manual, belum real-time.
- Poin bicara: sampaikan dengan tenang dan jujur — penguji lebih percaya pada kelompok yang tahu batas karyanya sendiri.

## Slide 12 — Kesimpulan, Saran & Penutup

- Kesimpulan: 2 sensor ponsel saja cukup untuk akurasi 95,33% pada data uji; suara penentu utama; web sudah berjalan dan teruji.
- Saran: tambah variasi ruangan, tambah fitur, sambungkan sensor real-time, bandingkan dengan Random Forest.
- Penutup: terima kasih, sebutkan link GitHub dan Colab, buka sesi tanya jawab.
- Poin bicara: tutup dengan satu kalimat — "RoomGuard menunjukkan bahwa solusi sederhana, kalau diuji dengan jujur, bisa bekerja dengan baik."

---

## Catatan untuk Presenter

- Bawa cadangan: screenshot halaman web dan tabel hasil, untuk jaga-jaga kalau demo live bermasalah (Wi-Fi, laptop, dan lain-lain).
- Bagi peran: satu orang menjelaskan data dan model (slide 4–9), satu orang memandu demo web (slide 10).
- Siapkan jawaban untuk pertanyaan yang paling mungkin muncul: kenapa hanya 2 fitur, kenapa akurasi privat lebih rendah, dan kenapa memilih Decision Tree bukan Random Forest.
