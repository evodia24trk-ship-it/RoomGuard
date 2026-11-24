# RoomGuard: Sistem Web Klasifikasi Keberadaan Orang dalam Ruangan Menggunakan Decision Tree Berbasis Sensor Ambience Smartphone

**Laporan Ujian Akhir Semester**
**Mata Kuliah: Workshop Machine Learning for IoT**

| | |
|---|---|
| Kelompok | 12 |
| Anggota | Evodia, Hasbi |
| Program | Workshop Machine Learning for IoT |
| Tanggal | November 2026 |

*Cara membaca laporan ini: setiap subbab dibuka dengan satu kalimat inti, supaya pembaca langsung tahu isi bagian itu.*

---

## BAB I — PENDAHULUAN

### 1.1 Latar Belakang

**Intinya:** lampu dan AC sering menyala di ruangan kosong, dan ponsel yang sudah ada di saku kita bisa membantu mendeteksinya.

Di gedung perkantoran dan kampus, lampu serta pendingin ruangan sering tetap menyala padahal tidak ada orang di dalamnya. Hal yang sama terjadi di rumah: lampu teras atau lampu kamar dibiarkan menyala semalaman. Kalau status ruangan bisa diketahui secara otomatis, perangkat itu bisa dimatikan tepat waktu dan energi yang terbuang bisa dikurangi.

Cara yang umum dipakai selama ini adalah memasang sensor khusus, misalnya sensor gerak PIR atau sensor CO₂. Cara ini bekerja, tetapi butuh biaya pemasangan dan perawatan. Tidak semua ruangan punya sensor seperti itu. Di sisi lain, hampir semua orang membawa ponsel pintar. Ponsel itu sudah punya sensor cahaya dan mikrofon bawaan. Pertanyaannya: apakah dua sinyal sederhana itu — seberapa terang dan seberapa berisik ruangan — sudah cukup untuk menebak ada orang atau tidak?

Penelitian sebelumnya menunjukkan jawabannya cenderung ya. Candanedo dan Feldheim (2016), misalnya, memakai data sensor suhu, kelembapan, cahaya, dan CO₂ untuk memperkirakan jumlah orang di ruangan, dan model mereka bekerja cukup baik. Bedanya, proyek ini jauh lebih sederhana: hanya dua fitur, yaitu intensitas cahaya (Lux) dan tingkat kebisingan (dB), dan sumber datanya ponsel biasa, bukan perangkat sensor khusus. Kalau dua fitur saja sudah memberi hasil yang memadai, sistem seperti ini bisa dibangun dengan murah dan mudah ditiru.

Proyek ini diberi nama **RoomGuard**. Sistemnya berupa aplikasi web: pengguna memasukkan nilai Lux dan dB, lalu sistem menjawab apakah ruangan Terisi atau Kosong, lengkap dengan rekomendasi hemat energi. Webnya ada dua wujud: versi lengkap berbasis Flask dan SQLite, serta demo publik di GitHub Pages yang berbahasa Indonesia dan bisa dibuka siapa saja. Di demo itu pengunjung juga bisa menambahkan data sendiri lewat fitur Tambah Dataset — penjelasannya ada di Bab III dan Bab IV. Di balik web itu ada model Decision Tree yang dilatih dari 2.000 baris data — 1.500 baris dari dataset publik UCI Room Occupancy Estimation Dataset dan 500 baris dari rekaman sensor ponsel anggota kelompok memakai aplikasi Phyphox. Perbandingan 75% data publik dan 25% data privat ini sengaja dipilih. Data publik memberi jumlah dan variasi yang cukup untuk melatih model, sedangkan data privat menguji apakah model yang dilatih dari data laboratorium tetap bekerja di ruangan nyata dengan ponsel biasa.

Decision Tree dipilih karena alasannya praktis. Hasil pelatihannya berupa aturan yang bisa dibaca manusia, misalnya "kalau suara di bawah sekian desibel, ruangan cenderung kosong". Sifat ini penting untuk laporan dan presentasi, karena penguji bisa melihat langsung logika yang dipakai model, bukan sekadar angka akurasi. Decision Tree juga ringan, cepat dilatih, dan cocok untuk data yang fiturnya sedikit.

### 1.2 Rumusan Masalah

**Intinya:** tiga pertanyaan yang dijawab proyek ini — bisa atau tidak, seberapa akurat, dan bagaimana menyajikannya di web.

1. Bagaimana membangun model klasifikasi yang membedakan ruangan Terisi dan Kosong hanya dari dua sinyal sensor ponsel, yaitu intensitas cahaya (Lux) dan tingkat kebisingan (dB)?
2. Seberapa akurat model Decision Tree yang dilatih dari gabungan dataset publik UCI dan data privat Phyphox, dan bagaimana kinerjanya jika diuji terpisah pada masing-masing sumber data?
3. Bagaimana menyajikan hasil klasifikasi itu dalam aplikasi web yang mudah dipakai, termasuk rekomendasi hemat energi, riwayat prediksi, dan cara bagi pengunjung untuk menambahkan data sendiri?

### 1.3 Batasan Masalah

**Intinya:** supaya selesai dalam satu semester, proyek ini dibatasi pada dua kelas, dua fitur, satu model, dan satu ruangan untuk data privat.

1. Klasifikasi hanya dua kelas: **Terisi** (ada orang di ruangan) dan **Kosong** (tidak ada orang). Jumlah orang tidak diperkirakan.
2. Fitur yang dipakai hanya dua: Light_Lux dan Sound_dB. Data suhu, kelembapan, dan CO₂ yang ada di dataset UCI tidak dipakai, karena ponsel biasa tidak selalu punya sensor itu, dan tujuan proyek ini adalah menguji seberapa jauh dua sensor ambience saja bisa membawa.
3. Dataset latih berjumlah 2.000 baris: 1.500 dari dataset publik UCI Room Occupancy Estimation Dataset dan 500 dari rekaman Phyphox anggota kelompok. Tidak ada penambahan data lain ke dalam dataset latih. Pengunjung web demo memang bisa menambahkan data sendiri lewat fitur Tambah Dataset, tetapi data itu hanya tersimpan di browser pengunjung dan tidak melatih ulang model — penjelasannya ada di Bab III.
4. Model yang dipakai hanya Decision Tree dari pustaka scikit-learn. Model lain tidak dibandingkan secara mendalam dalam laporan ini.
5. Data privat diambil di dalam ruangan tertutup dengan ponsel diletakkan di atas meja, interval pencatatan 30 detik, dan pelabelan manual berdasarkan lembar observasi (ada atau tidak ada orang di ruangan), bukan dari nilai sensornya.
6. Aplikasi web punya dua wujud: versi lengkap berbasis Flask dan SQLite yang dijalankan secara lokal, dan demo statis berbahasa Indonesia di GitHub Pages yang modelnya berjalan langsung di browser. Keduanya ditujukan untuk demonstrasi dan pengujian, bukan untuk penggunaan produksi.

### 1.4 Tujuan

**Intinya:** membangun modelnya, menguji dengan data gabungan, lalu menampilkannya di web.

1. Mengembangkan model Decision Tree yang mengklasifikasikan status keberadaan orang dalam ruangan (Terisi atau Kosong) berdasarkan intensitas cahaya dan tingkat kebisingan dari sensor ponsel.
2. Menggabungkan dataset publik UCI dan dataset privat Phyphox menjadi satu dataset latih sebanyak 2.000 baris, lalu mengevaluasi kinerja model pada seluruh data uji maupun terpisah per sumber data.
3. Membangun aplikasi web RoomGuard yang menampilkan hasil prediksi, rekomendasi hemat energi, riwayat prediksi, dan fitur Tambah Dataset, sehingga hasil model bisa dipakai dan diuji secara langsung, dan pengunjung bisa ikut menambahkan data.

### 1.5 Manfaat

**Intinya:** pengguna bisa menghemat energi, mahasiswa lain bisa meniru alurnya, dan peneliti lanjutan punya titik awal yang jelas.

1. **Bagi pengguna ruangan**, sistem ini memberi gambaran sederhana tentang bagaimana status ruangan bisa dipantau tanpa memasang perangkat khusus, dan membantu kebiasaan hemat energi lewat rekomendasi di web.
2. **Bagi mahasiswa dan pengembang**, proyek ini menunjukkan alur kerja machine learning yang lengkap — dari pengumpulan data dengan ponsel, pembersihan dan penggabungan data, pelatihan model, sampai penyajian lewat aplikasi web — dengan dua fitur saja.
3. **Bagi penelitian lanjutan**, hasil proyek ini, termasuk kelemahannya, bisa menjadi titik awal untuk menambah sensor, memakai model lain, atau menguji di ruangan yang lebih beragam.

---

## BAB II — TINJAUAN PUSTAKA

### 2.1 Penelitian Terdahulu

**Intinya:** cahaya dan suara memang terbukti informatif, tetapi penelitian yang hanya memakai satu jenis sensor biasanya mencatat akurasi yang lebih rendah.

Topik pendeteksian keberadaan orang di ruangan sudah diteliti cukup lama, terutama untuk penghematan energi di gedung. Tabel 2.1 merangkum empat penelitian yang relevan. Angka-angka pada tabel diambil dari publikasi masing-masing penelitian. Dataset dan kondisi pengujiannya berbeda, jadi angka itu tidak bisa dibandingkan langsung satu sama lain. Namun arah temuannya konsisten: sinyal lingkungan seperti cahaya dan suara membawa informasi yang berguna tentang ada tidaknya orang di ruangan.

**Tabel 2.1 — Penelitian terdahulu yang relevan**

| No | Penelitian (tahun) | Dataset / sumber data | Metode | Hasil yang dilaporkan | Keterbatasan |
|---|---|---|---|---|---|
| 1 | Candanedo & Feldheim (2016) | Data sensor ruang kantor: suhu, kelembapan, cahaya, CO₂, dan rasio kelembapan; 8.144 baris | Random Forest, Gradient Boosting, dan LDA | Akurasi sekitar 95–99% pada klasifikasi biner terisi/kosong | Memakai sensor khusus (termasuk CO₂) yang tidak ada di semua ponsel; pengujian di satu ruangan kantor |
| 2 | Akkaya dkk. (2015) | Data sensor nirkabel di gedung perkantoran, termasuk PIR dan sensor lingkungan | Decision Tree dan k-NN | Akurasi sekitar 90% dengan kombinasi beberapa jenis sensor | Membutuhkan pemasangan jaringan sensor; kinerja menurun ketika hanya satu jenis sensor dipakai |
| 3 | D. Wang dkk. (2017), occupancy estimation berbasis suara | Rekaman audio ruangan dan fitur spektral suara | SVM dan klasifikasi berbasis fitur audio | Akurasi sekitar 85–92% untuk membedakan ruangan terisi dan kosong dari suara | Sensitif terhadap kebisingan dari luar ruangan; privasi rekaman audio menjadi perhatian |
| 4 | H. Zou dkk. (2018) | Data sensor cahaya dan gerak dari beberapa ruangan | Logistic Regression dan Decision Tree | Akurasi sekitar 88–93% untuk klasifikasi biner | Kinerja bergantung pada posisi sensor dan pola pencahayaan ruangan; jumlah ruangan terbatas |

Ada dua pelajaran yang dipakai dalam proyek ini. Pertama, cahaya dan suara memang termasuk sinyal yang paling informatif untuk soal ini, jadi masuk akal untuk menguji keduanya saja tanpa sensor tambahan. Kedua, penelitian yang hanya mengandalkan satu jenis sensor cenderung mencatat akurasi yang lebih rendah dibanding yang memakai banyak sensor. Proyek ini menerima konsekuensi itu. Targetnya bukan mengalahkan penelitian dengan sensor lengkap, melainkan membuktikan bahwa dua sensor ponsel saja sudah cukup baik untuk keperluan praktis sederhana, dengan biaya dan kerumitan yang jauh lebih rendah.

### 2.2 Decision Tree dan Klasifikasi Biner

**Intinya:** Decision Tree bekerja seperti diagram alir — setiap cabangnya mengajukan satu pertanyaan, dan jawabannya bisa diikuti siapa pun.

Decision Tree membagi data secara bertahap berdasarkan nilai fitur. Setiap titik percabangan (node) mengajukan satu pertanyaan, misalnya "apakah Sound_dB lebih besar dari 46,35?", lalu data dialirkan ke cabang kiri atau kanan sesuai jawabannya. Proses ini berulang sampai mencapai daun (leaf) yang berisi keputusan kelas. Karena strukturnya seperti diagram alir, orang tanpa latar belakang machine learning pun bisa mengikuti logika modelnya. Inilah alasan utama Decision Tree dipilih untuk proyek ini.

Untuk klasifikasi biner — hanya dua kelas, Terisi dan Kosong — Decision Tree memilih pemisahan yang paling mengurangi ketidakpastian pada setiap node. Proyek ini memakai kriteria *gini impurity*: sebuah pemisahan dianggap baik kalau kelompok hasil pemisahan itu semakin homogen, yakni semakin banyak berisi satu kelas saja. Dua pengaturan yang cukup berpengaruh adalah `max_depth` (seberapa dalam pohon boleh tumbuh) dan `min_samples_leaf` (jumlah minimum data pada setiap daun). Pohon yang terlalu dalam akan menghafal data latih dan gagal pada data baru; pohon yang terlalu dangkal tidak cukup menangkap pola. Karena itu kedua nilai tersebut tidak ditebak, melainkan dicari lewat GridSearchCV (lihat Bab III).

Kelebihan Decision Tree adalah kemudahan interpretasi dan kecepatan pelatihan. Kelemahannya, pohon tunggal bisa tidak stabil: perubahan kecil pada data latih bisa mengubah struktur pohon. Untuk proyek ini kelemahan itu masih bisa diterima, karena tujuannya adalah model yang bisa dijelaskan dan diuji, bukan model dengan akurasi tertinggi yang bisa dicapai.

### 2.3 Preprocessing Sensor

**Intinya:** empat kolom cahaya dan empat kolom suara diringkas menjadi satu nilai cahaya dan satu nilai suara per baris, dan label privat tidak boleh ditentukan dari nilai sensor.

Data mentah dari sensor jarang bisa langsung dipakai. Pada dataset UCI, nilai cahaya ada dalam empat kolom (S1_Light sampai S4_Light) dan nilai suara dalam empat kolom (S1_Sound sampai S4_Sound), dari empat titik sensor di ruangan. Proyek ini mereduksinya menjadi satu nilai cahaya dan satu nilai suara per baris. Cahaya diambil dari nilai maksimum keempat sensor. Suara dihitung dari nilai maksimum keempat sensor suara dengan rumus `20 × log10(maksimum) + 60`, sehingga satuannya menjadi desibel. Nilai maksimum dipilih karena paling mewakili kondisi ruangan: kalau ada satu titik yang terang atau berisik, ruangan itu kemungkinan memang sedang dipakai.

Untuk data privat dari Phyphox, langkahnya lebih sederhana karena aplikasi itu sudah mencatat intensitas cahaya dalam Lux dan tingkat tekanan suara dalam dB. Yang perlu dilakukan adalah menyamakan format kolom dan rentang nilainya dengan data publik. Setelah penggabungan, kedua sumber diperiksa dengan cara yang sama: nilai di luar rentang wajar (Lux 0–280, dB 25–85) diperiksa ulang, baris dengan nilai kosong dibuang, dan label dipastikan hanya berisi dua nilai, Terisi atau Kosong.

Satu hal yang dijaga dengan ketat: pelabelan data privat tidak boleh didasarkan pada nilai sensor. Label ditentukan dari lembar observasi — pencatat menulis apakah ada orang di ruangan pada saat itu atau tidak — dan baru sesudah itu dicocokkan dengan baris sensor pada waktu yang sama. Kalau label ditentukan dari nilai sensornya sendiri, model hanya akan belajar meniru aturan pelabelan itu, dan hasil evaluasinya tidak lagi bermakna.

### 2.4 Frontend, Backend, dan API

**Intinya:** ada dua wujud web — demo publik berbahasa Indonesia dan versi lengkap Flask berbahasa Inggris — dan keduanya memakai model yang sama.

**Demo publik.** Demo ini tampil di GitHub Pages dan bisa dibuka siapa saja tanpa instalasi. Seluruh tampilannya berbahasa Indonesia dan dibuat sederhana: latar krem, header hijau tua, dan kartu putih. Modelnya berjalan langsung di browser, mengikuti aturan Decision Tree yang sama dengan versi lengkap. Halaman ini terdiri dari lima bagian: Coba klasifikasi, Riwayat percobaan, Seberapa bagus modelnya, Datanya dari mana, dan Tambah Dataset. Di bagian Coba klasifikasi, pengunjung memasukkan nilai Lux dan dB, lalu hasilnya muncul beserta alasan singkat dan saran hemat energi. Riwayat percobaan tersimpan di browser pengunjung sendiri.

Bagian **Tambah Dataset** adalah tambahan terbaru pada demo ini. Pengunjung bisa menambahkan data sendiri dengan dua cara: mengisi form satu baris (Cahaya, Suara, dan Label Terisi atau Kosong), atau mengunggah file CSV dengan kolom Light_Lux, Sound_dB, dan Occupancy. Data tambahan tampil dalam tabel, bisa dihapus per baris, dan bisa diunduh kembali sebagai file CSV. Penghitung di bagian atas menunjukkan jumlah data tambahan dan totalnya, misalnya dataset awal 2.000 baris ditambah 3 baris data tambahan menjadi 2.003 baris. Semua data tambahan tersimpan di browser pengunjung (localStorage), sehingga tidak hilang saat halaman dimuat ulang. Perlu ditegaskan di sini: data tambahan itu belum melatih ulang model. Model yang dipakai demo tetap model yang dilatih dari 2.000 baris data awal; untuk melatih ulang dengan data baru, langkahnya ada di notebook Colab.

**Versi lengkap.** Versi lengkap dibangun dengan Flask dan SQLite, dan seluruh tampilannya berbahasa Inggris. Menunya terdiri dari Dashboard, Predict, History, Dataset, Model, dan About. Dashboard menampilkan ringkasan dan prediksi terakhir. Predict adalah halaman untuk memasukkan nilai Lux dan dB. History menampilkan riwayat prediksi yang tersimpan di database, lengkap dengan fitur ubah dan hapus. Dataset menampilkan ringkasan data. Model menampilkan informasi model dan kinerjanya. About menjelaskan proyek ini.

Backend memakai Flask, framework web Python yang ringan dan cocok untuk proyek skala kecil. Backend menerima input dari frontend, memanggil model Decision Tree yang sudah dilatih, menyimpan hasil prediksi ke database SQLite, dan mengembalikan jawaban dalam format JSON melalui endpoint API. SQLite dipilih karena tidak butuh instalasi server database terpisah. Seluruh data tersimpan dalam satu file, sehingga aplikasi mudah dijalankan di laptop mana pun untuk keperluan demonstrasi.

### 2.5 Evaluasi Model dan Aplikasi

**Intinya:** akurasi saja tidak cukup — presisi, recall, F1-score, dan confusion matrix dibaca bersama-sama, dan model juga diuji terpisah pada data publik dan privat.

Akurasi adalah persentase prediksi yang benar dari seluruh data uji. Presisi menjawab pertanyaan: dari semua yang diprediksi Terisi, berapa persen yang memang Terisi. Recall menjawab pertanyaan sebaliknya: dari semua ruangan yang memang Terisi, berapa persen yang berhasil dikenali. F1-score adalah rata-rata harmonis dari presisi dan recall, berguna ketika kedua ukuran itu sama pentingnya. Keempat ukuran ini dibaca bersama confusion matrix, tabel yang menunjukkan berapa prediksi benar dan salah untuk masing-masing kelas.

Model juga dievaluasi terpisah pada data uji publik dan privat. Pemisahan ini penting karena kedua sumber data berasal dari kondisi yang berbeda: data publik direkam dengan sensor khusus di ruangan kantor, sedangkan data privat direkam dengan ponsel di ruangan biasa. Kalau model bekerja jauh lebih buruk pada data privat, itu pertanda model terlalu menyesuaikan diri dengan karakteristik data publik.

Evaluasi aplikasi dilakukan secara fungsional: setiap menu dan tombol diuji apakah berjalan sebagaimana mestinya, prediksi tersimpan atau tidak, dan apakah rekomendasi hemat energi yang muncul sesuai dengan hasil prediksinya. Rencana pengujian dan kriteria kelulusannya dijelaskan pada Bab III.

---

## BAB III — METODOLOGI

### 3.1 Tahapan Pengerjaan

**Intinya:** delapan langkah berurutan — dari pengumpulan data sampai pengujian dan pelaporan.

Pengerjaan proyek ini mengikuti alur kerja machine learning yang umum: mulai dari pengumpulan data, pembersihan dan penggabungan, pembagian data, pelatihan dan pemilihan model, evaluasi, sampai penyajian lewat aplikasi web. Gambar 3.1 menunjukkan alur tersebut.

**Gambar 3.1 — Flowchart tahapan pengerjaan (dalam Bahasa Inggris)**

```
┌─────────────────────────────────────────────┐
│  START                                      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  1. DATA COLLECTION                         │
│  • Public: UCI Room Occupancy Estimation    │
│    Dataset #864 (10,129 raw rows)           │
│  • Private: Phyphox light + microphone      │
│    recordings, 30-second interval,          │
│    5 sessions, manual labelling             │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  2. DATA CHECKING & CLEANING                │
│  • Check missing values and ranges          │
│    (Light 0–280 Lux, Sound 25–85 dB)        │
│  • Derive label: count > 0 → Occupied      │
│  • Merge into 2,000 rows                    │
│    (1,500 public + 500 private)             │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  3. FEATURE ENGINEERING                     │
│  • Light_Lux = max(S1–S4_Light)             │
│  • Sound_dB = 20*log10(max(S1–S4_Sound))    │
│    + 60                                     │
│  • Final features: Light_Lux, Sound_dB      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  4. DATA SPLITTING (stratified, seed 42)    │
│  • Train 70% (1,400)                        │
│  • Validation 15% (300)                     │
│  • Test 15% (300)                           │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  5. MODEL TRAINING                          │
│  • Decision Tree + GridSearchCV (5-fold)    │
│  • Best: gini, max_depth = 5,               │
│    min_samples_leaf = 10                    │
│  • CV accuracy = 97.29%                     │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  6. MODEL EVALUATION                        │
│  • Accuracy, precision, recall, F1-score    │
│  • Confusion matrix on test set (n = 300)   │
│  • Per-source evaluation: public vs private │
│  • Feature importance analysis              │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  7. WEB APPLICATION                         │
│  • Flask backend + SQLite database          │
│  • Pages: Dashboard, Predict, History,      │
│    Dataset, Model, About                    │
│  • Energy-saving recommendation             │
│  • Public demo (GitHub Pages, Indonesian)   │
│    with Tambah Dataset feature              │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  8. TESTING & REPORTING                     │
│  • Functional testing of all pages          │
│  • Documentation and presentation           │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│  END                                        │
└─────────────────────────────────────────────┘
```

### 3.2 Sumber dan Cara Mendapatkan Data

**Intinya:** data publik diambil dari UCI lalu diseimbangkan menjadi 750 Terisi dan 750 Kosong; data privat direkam sendiri dengan Phyphox dalam 5 sesi.

**Data publik.** Dataset yang dipakai adalah UCI Room Occupancy Estimation Dataset (nomor 864) dari UCI Machine Learning Repository, diakses pada 24 November 2026 melalui https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation. File mentahnya berisi 10.129 baris dengan kolom tanggal, jumlah orang (`count`), empat kolom cahaya (S1_Light sampai S4_Light), dan empat kolom suara (S1_Sound sampai S4_Sound). Dari jumlah itu, 8.228 baris berlabel Kosong dan 1.901 baris berlabel Terisi. Ketidakseimbangan ini wajar untuk data ruangan kantor: sebagian besar waktu ruangan memang kosong.

Karena proyek ini memakai 1.500 baris publik, sampel diambil secara berstrata dan seimbang: 750 Terisi dan 750 Kosong. Pilihan ini sengaja dibuat tidak mengikuti proporsi asli data mentah (81% Kosong), supaya model tidak belajar menebak kelas mayoritas saja. Dengan jumlah kedua kelas yang sama, akurasi menjadi ukuran yang lebih adil.

**Data privat.** Data privat direkam sendiri oleh anggota kelompok menggunakan aplikasi Phyphox di ponsel pintar. Ponsel diletakkan di atas meja di dalam ruangan, lalu Phyphox mencatat intensitas cahaya dan tingkat suara setiap 30 detik. Pengambilan dilakukan dalam 5 sesi yang berbeda waktu, dengan total 500 baris: 250 baris saat ruangan Terisi dan 250 baris saat ruangan Kosong. Jumlah kedua kelas sengaja dibuat sama supaya data privat seimbang.

Setiap sesi berjalan seperti ini. Pertama, pencatat menyiapkan lembar observasi sederhana berisi waktu dan kondisi ruangan. Kedua, ponsel diletakkan di tempat yang sama pada setiap sesi, layarnya menghadap ke atas agar sensor cahaya tidak tertutup. Ketiga, Phyphox dijalankan dan dibiarkan merekam selama sesi berlangsung. Keempat, pencatat menulis pada lembar observasi kapan ada orang masuk atau keluar ruangan. Kelima, setelah sesi selesai, data diekspor dari Phyphox dalam format CSV dan dicocokkan dengan lembar observasi berdasarkan waktu pencatatan.

Aturan pelabelan data privat perlu ditegaskan di sini. Label **Terisi** diberikan kalau pada waktu pencatatan itu memang ada minimal satu orang di dalam ruangan menurut lembar observasi, dan **Kosong** kalau tidak ada orang sama sekali. Label ditentukan dari observasi, bukan dari nilai sensor. Artinya, kalau ruangan sedang sepi tetapi lampunya menyala terang, baris itu tetap berlabel Kosong. Kalau ruangan sedang terisi tetapi suasananya hening, baris itu tetap berlabel Terisi. Aturan ini penting supaya model diuji secara jujur: model harus belajar dari pola sensor yang sesungguhnya, bukan dari label yang sudah "dibocorkan" oleh nilai sensor itu sendiri.

### 3.3 Pemeriksaan dan Pembagian Data

**Intinya:** data dibersihkan dulu, lalu digabung menjadi 2.000 baris dan dibagi 70/15/15 dengan proporsi kelas yang sama di setiap bagian.

Sebelum digabung, kedua sumber data diperiksa. Pemeriksaan meliputi: apakah ada baris dengan nilai kosong, apakah ada nilai di luar rentang wajar (Lux 0–280, dB 25–85), dan apakah ada baris duplikat. Baris dengan nilai kosong dibuang, sedangkan nilai di luar rentang diperiksa apakah berasal dari kesalahan pencatatan atau memang kondisi ruangan yang tidak biasa.

Setelah bersih, data digabung menjadi 2.000 baris (1.500 publik dan 500 privat). Gabungan ini kemudian dibagi dengan metode *stratified split* dan `random_state` 42, dengan proporsi 70% latih, 15% validasi, dan 15% uji. Pembagian berstrata artinya proporsi kelas Terisi dan Kosong dijaga sama di setiap bagian, sehingga tidak ada bagian yang kebetulan berisi satu kelas saja. Hasil pembagiannya: train 1.400 baris, validation 300 baris, dan test 300 baris, masing-masing seimbang antara kedua kelas. Nilai seed 42 dipakai supaya pembagian ini bisa diulang persis sama oleh siapa pun yang menjalankan kode yang sama.

### 3.4 Preprocessing

**Intinya:** lima langkah — hitung cahaya, hitung suara, tentukan label, samakan format, lalu periksa lagi. Tanpa normalisasi, karena Decision Tree tidak membutuhkannya.

1. **Menghitung fitur cahaya.** `Light_Lux` diambil dari nilai maksimum empat sensor cahaya pada data publik. Pada data privat, nilai Lux dari Phyphox dipakai langsung.
2. **Menghitung fitur suara.** `Sound_dB` dihitung dari nilai maksimum empat sensor suara pada data publik dengan rumus `20 × log10(maksimum) + 60`. Pada data privat, nilai dB dari Phyphox dipakai langsung.
3. **Menentukan label.** Pada data publik, label Terisi diberikan kalau kolom `count` lebih besar dari nol. Pada data privat, label diambil dari lembar observasi seperti dijelaskan pada bagian 3.2.
4. **Menyamakan format.** Kedua sumber disatukan ke dalam satu tabel dengan kolom yang sama: Light_Lux, Sound_dB, dan label.
5. **Pemeriksaan akhir.** Rentang nilai diperiksa kembali (Lux 0–280, dB 25–85), lalu data dibagi seperti pada bagian 3.3.

Tidak dilakukan normalisasi atau standardisasi, karena Decision Tree tidak sensitif terhadap skala fitur: pohon hanya membandingkan nilai fitur dengan ambang batas, sehingga mengubah skala tidak mengubah hasil pemisahannya.

### 3.5 Arsitektur dan Pelatihan Decision Tree

**Intinya:** dua fitur masuk, satu dari dua kelas keluar; pengaturan terbaik dicari dengan GridSearchCV, bukan ditebak.

Model yang dipakai adalah `DecisionTreeClassifier` dari scikit-learn. Arsitekturnya sederhana: dua fitur masukan (Light_Lux dan Sound_dB) masuk ke pohon keputusan, dan keluarannya adalah satu dari dua kelas, Terisi atau Kosong.

Pelatihan dilakukan dalam dua langkah. Pertama, pencarian hyperparameter dengan GridSearchCV dan validasi silang 5-fold pada data latih. Grid yang dicoba mencakup kriteria pemisahan (gini dan entropy), kedalaman maksimum (3, 5, 7, dan tanpa batas), serta jumlah minimum sampel per daun (1, 5, dan 10). Kombinasi terbaik yang ditemukan adalah kriteria **gini**, `max_depth` **5**, dan `min_samples_leaf` **10**, dengan akurasi validasi silang **97,29%**. Kedua, model dengan pengaturan terbaik itu dilatih ulang pada seluruh data latih, lalu dievaluasi pada data validasi dan data uji.

Kedalaman 5 dipilih bukan karena paling dalam, melainkan karena paling baik pada validasi silang. Pohon yang lebih dalam memang menaikkan akurasi latih, tetapi tidak lagi memperbaiki akurasi validasi — tanda bahwa pohon mulai menghafal data latih. Nilai `min_samples_leaf` 10 juga berperan sebagai pengendali: setiap daun harus berisi minimal 10 data, sehingga pohon tidak membuat aturan berdasarkan segelintir baris saja.

Seluruh eksperimen dijalankan di Google Colab dengan Python dan scikit-learn, supaya langkah-langkahnya terdokumentasi dalam satu notebook dan bisa dijalankan ulang.

### 3.6 Perancangan Frontend, Backend, dan API

**Intinya:** enam halaman di depan untuk versi lengkap, Flask dan SQLite di belakang, dihubungkan lewat endpoint API — ditambah demo publik satu halaman dengan fitur Tambah Dataset.

**Versi lengkap (Flask).** Enam halaman utama: Dashboard (ringkasan, prediksi terakhir, dan statistik singkat), Predict (form input Lux dan dB beserta hasil prediksi dan rekomendasi hemat energi), History (daftar prediksi yang tersimpan, dengan fitur ubah dan hapus), Dataset (ringkasan data dan komposisi sumber), Model (informasi model, hyperparameter, dan metrik kinerja), dan About (penjelasan proyek dan anggota kelompok). Tampilan dibuat responsif dan seluruhnya berbahasa Inggris.

**Backend dan API.** Backend memakai Flask dengan endpoint sebagai berikut:

| Endpoint | Metode | Fungsi |
|---|---|---|
| `/` | GET | Halaman Dashboard |
| `/predict` | GET / POST | Halaman dan proses prediksi |
| `/api/predict` | POST | Menerima JSON berisi Lux dan dB, mengembalikan kelas prediksi, probabilitas, dan rekomendasi |
| `/history` | GET | Menampilkan riwayat prediksi |
| `/api/history` | GET / DELETE | Membaca dan menghapus riwayat |
| `/dataset` | GET | Ringkasan dataset |
| `/model` | GET | Informasi model dan metrik |
| `/about` | GET | Halaman tentang proyek |

Hasil prediksi disimpan ke SQLite, termasuk nilai input, kelas hasil, probabilitas, dan waktu prediksi. Rekomendasi hemat energi dibuat berdasarkan hasil prediksi: kalau ruangan diprediksi Kosong, sistem menyarankan mematikan lampu dan pendingin ruangan; kalau Terisi, sistem menyarankan pengaturan normal dan mengingatkan untuk mematikan perangkat saat ruangan ditinggalkan.

**Demo publik (GitHub Pages).** Demo ini adalah satu halaman berbahasa Indonesia dengan tampilan sederhana: latar krem, header hijau tua, dan kartu putih. Aturan Decision Tree yang dipakai sama dengan versi lengkap, hanya saja dijalankan langsung di browser dengan JavaScript, tanpa server. Urutan bagiannya adalah Coba klasifikasi, Riwayat percobaan, Seberapa bagus modelnya, Datanya dari mana, dan Tambah Dataset.

Fitur Tambah Dataset dirancang supaya pengunjung bisa ikut menambahkan data. Ada form untuk satu baris data (Cahaya, Suara, Label), tombol unggah file CSV dengan kolom Light_Lux, Sound_dB, dan Occupancy (nilai Occupancy boleh 1/0 atau Terisi/Kosong), tabel data tambahan dengan tombol hapus per baris, tombol unduh CSV, dan penghitung total yang berubah otomatis. Validasi inputnya sama dengan form klasifikasi: Cahaya harus 0–600 Lux dan Suara 25–85 dB, kalau tidak, muncul pesan galat berbahasa Indonesia dan barisnya tidak ditambahkan. Baris CSV yang tidak valid dilewati dan dilaporkan jumlahnya. Semua data tambahan disimpan di browser pengunjung (localStorage) dan tidak dikirim ke mana-mana. Demo ini juga menulis dengan jujur di halamannya bahwa data tambahan belum melatih ulang model.

**Wireframe (deskripsi).** Pada versi lengkap, halaman Predict menampilkan dua kolom input angka (Light dalam Lux dan Sound dalam dB) di sebelah kiri, tombol Predict di bawahnya, dan kartu hasil di sebelah kanan yang menunjukkan label Terisi atau Kosong, persentase keyakinan, serta kotak rekomendasi. Dashboard menampilkan empat kartu statistik di bagian atas (total prediksi, persentase Terisi, persentase Kosong, dan akurasi model), diikuti tabel lima prediksi terakhir. History menampilkan tabel riwayat dengan kolom waktu, Lux, dB, hasil, dan aksi hapus. Halaman Model menampilkan metrik kinerja dalam kartu-kartu kecil dan confusion matrix dalam bentuk tabel. Pada demo publik, semua bagian tersusun dalam satu halaman yang bisa di-scroll dari atas ke bawah, dan bagian Tambah Dataset terletak paling bawah, setelah bagian Datanya dari mana.

### 3.7 Rencana Pengujian dan Kriteria Kelulusan

**Intinya:** model lulus kalau akurasinya minimal 90% dan tidak terlalu bergantung pada satu sumber data; aplikasi lulus kalau semua halaman dan tombol berfungsi.

**Pengujian model.** Model dinyatakan lulus kalau memenuhi kriteria berikut: akurasi pada data uji minimal 90%; presisi dan recall masing-masing kelas minimal 85%; dan selisih akurasi antara data uji publik dan privat tidak lebih dari 10 poin persentase. Selisih yang lebih besar menandakan model terlalu bergantung pada satu sumber data.

**Pengujian aplikasi.** Setiap halaman dibuka dan setiap tombol dicoba: form prediksi harus menerima input dan menampilkan hasil; riwayat harus tersimpan dan bisa dihapus; rekomendasi harus sesuai dengan hasil prediksi; dan aplikasi harus tetap berjalan tanpa error saat diberi input di luar rentang wajar (misalnya Lux negatif atau dB di atas 85). Pengujian dilakukan secara manual oleh anggota kelompok dengan daftar periksa (checklist), dan hasilnya dicatat pada Bab IV.

**Pengujian web demo.** Pada demo publik, fitur Tambah Dataset diuji dengan empat langkah: menambahkan satu baris lewat form, mengunggah file CSV berisi baris valid dan satu baris di luar rentang, menghapus satu baris dari tabel, dan mengunduh data tambahan sebagai CSV. Kriteria lulusnya: penghitung total berubah sesuai jumlah data tambahan, baris yang ditambahkan muncul di tabel, baris CSV yang tidak valid dilewati dan dilaporkan, file unduhan berisi data yang sama dengan tabel, dan data tambahan tetap tersimpan setelah halaman dimuat ulang.

---

## BAB IV — HASIL DAN PEMBAHASAN

### 4.1 Dataset

**Intinya:** dataset akhir 2.000 baris, seimbang, dan terbagi rapi menjadi latih 1.400, validasi 300, dan uji 300.

Dataset akhir berjumlah **2.000 baris**, terdiri dari **1.500 baris publik** (75%) dari UCI Room Occupancy Estimation Dataset dan **500 baris privat** (25%) dari rekaman Phyphox. File mentah dataset UCI sendiri berisi 10.129 baris (8.228 Kosong dan 1.901 Terisi); dari jumlah itu diambil 1.500 baris sebagai porsi publik proyek ini. Data privat dikumpulkan dalam 5 sesi dengan interval 30 detik, seimbang antara 250 baris Terisi dan 250 baris Kosong.

Tabel 4.1 merangkum komposisi dataset setelah pembagian 70/15/15 secara berstrata dengan seed 42.

**Tabel 4.1 — Komposisi dataset setelah pembagian**

| Bagian | Jumlah | Terisi | Kosong |
|---|---|---|---|
| Train (70%) | 1.400 | 700 | 700 |
| Validation (15%) | 300 | 150 | 150 |
| Test (15%) | 300 | 150 | 150 |
| **Total** | **2.000** | **1.000** | **1.000** |

Dua fitur yang dipakai adalah Light_Lux (rentang 0–280) dan Sound_dB (rentang 25–85 dB). Dataset gabungan ini seimbang (1.000 Terisi dan 1.000 Kosong), sehingga akurasi menjadi ukuran yang cukup adil: model tidak bisa mendapat nilai tinggi hanya dengan selalu menebak satu kelas.

### 4.2 Hasil Pelatihan

**Intinya:** pengaturan terbaik adalah gini, kedalaman 5, dan daun minimal 10 — dan suara jauh lebih menentukan daripada cahaya.

Pencarian hyperparameter dengan GridSearchCV (5-fold) menghasilkan kombinasi terbaik: kriteria **gini**, `max_depth` **5**, dan `min_samples_leaf` **10**, dengan akurasi validasi silang **97,29%**. Model dengan pengaturan ini kemudian dilatih pada seluruh data latih dan dievaluasi pada data validasi dan uji.

**Tabel 4.2 — Akurasi pada tiap bagian data**

| Bagian | Jumlah data | Akurasi |
|---|---|---|
| Train | 1.400 | 97,86% |
| Validation | 300 | 95,00% |
| Test | 300 | 95,33% |

Selisih antara akurasi latih (97,86%) dan akurasi uji (95,33%) hanya sekitar 2,5 poin persentase. Selisih yang kecil ini pertanda baik: model tidak menghafal data latih secara berlebihan. Kalau selisihnya besar — misalnya latih 99% tetapi uji 80% — itu baru patut dicurigai sebagai overfitting. Di sini tidak terjadi.

**Feature importance.** Dari dua fitur yang dipakai, Sound_dB jauh lebih berpengaruh: nilai importance-nya **0,9006**, sedangkan Light_Lux hanya **0,0994**. Dengan kata lain, sekitar 90% keputusan pohon didorong oleh tingkat kebisingan. Hal ini masuk akal: ruangan yang terisi hampir selalu menimbulkan suara — langkah kaki, percakapan, bunyi keyboard — sedangkan ruangan kosong cenderung hening. Cahaya membantu, tetapi sifatnya pelengkap: lampu bisa saja menyala di ruangan kosong, dan ruangan terisi pun bisa saja remang. Karena itu model menempatkan suara sebagai penentu utama dan cahaya sebagai penyeimbang pada kasus yang meragukan.

Aturan utama yang dipelajari pohon juga sejalan dengan temuan ini. Secara umum, **Sound_dB di bawah atau sama dengan 41,5 cenderung mengarah ke Kosong**, sedangkan **Sound_dB di atas 46,35 cenderung mengarah ke Terisi**. Pada rentang di antaranya (41,5 sampai 46,35 dB), model memakai Light_Lux untuk membantu memutuskan. Aturan yang sederhana dan mudah dibaca seperti ini adalah salah satu kelebihan Decision Tree: penguji bisa melihat langsung logika yang dipakai, bukan sekadar percaya pada angka.

<!-- SCREENSHOT: models/decision_tree.png — Visualisasi Decision Tree -->

### 4.3 Confusion Matrix dan Analisis Kesalahan

**Intinya:** dari 300 data uji, 286 benar dan 14 salah — 6 ruangan kosong dikira terisi, 8 ruangan terisi dikira kosong.

Pada data uji (n = 300), model mencapai **akurasi 95,33%**, **presisi 95,95%**, **recall 94,67%**, dan **F1-score 95,30%**. Per kelas, hasilnya adalah sebagai berikut.

**Tabel 4.3 — Metrik per kelas pada data uji**

| Kelas | Presisi | Recall | F1-score |
|---|---|---|---|
| Kosong | 0,95 | 0,96 | 0,95 |
| Terisi | 0,96 | 0,95 | 0,95 |

Confusion matrix-nya adalah sebagai berikut (baris = kondisi sebenarnya, kolom = prediksi model):

**Tabel 4.4 — Confusion matrix pada data uji (n = 300)**

|  | Diprediksi Kosong | Diprediksi Terisi |
|---|---|---|
| **Sebenarnya Kosong** | 144 (benar) | 6 (salah) |
| **Sebenarnya Terisi** | 8 (salah) | 142 (benar) |

<!-- SCREENSHOT: models/confusion_matrix.png — Confusion matrix data uji -->

Dari 300 data uji, 286 diprediksi dengan benar dan 14 salah. Ke-14 kesalahan itu terbagi menjadi 6 false positive (ruangan sebenarnya Kosong tetapi diprediksi Terisi) dan 8 false negative (ruangan sebenarnya Terisi tetapi diprediksi Kosong).

Kedua jenis kesalahan ini punya penyebab yang bisa dijelaskan. **False positive (6 kasus)** kemungkinan besar berasal dari ruangan kosong yang kebisingannya tinggi: suara dari koridor, ruangan sebelah, atau perangkat yang menyala bisa membuat nilai dB melewati ambang batas, sehingga model mengira ada orang. Lampu yang menyala di ruangan kosong juga memperkuat dugaan yang salah itu. **False negative (8 kasus)** sebaliknya: ruangan terisi tetapi suasananya hening — misalnya satu orang yang sedang membaca tanpa bersuara — sehingga nilai dB-nya rendah dan model mengira ruangan kosong. Cahaya yang redup pada saat itu membuat model semakin yakin pada jawaban yang salah.

Dampak kedua kesalahan ini berbeda. False positive membuat sistem mengira ruangan terisi padahal kosong, sehingga lampu dan pendingin tetap menyala: kerugiannya adalah energi yang terbuang. False negative membuat sistem mengira ruangan kosong padahal terisi, sehingga perangkat bisa dimatikan saat masih ada orang: kerugiannya adalah kenyamanan pengguna. Dalam proyek ini jumlah keduanya hampir seimbang (6 lawan 8), dan tidak ada jenis kesalahan yang mendominasi, sehingga model bisa dikatakan cukup adil terhadap kedua kelas.

**Kinerja per sumber data.** Ketika data uji dipecah berdasarkan asalnya, model mencatat akurasi **96,94%** pada data uji publik (n = 229) dan **90,14%** pada data uji privat (n = 71). Selisih sekitar 6,8 poin persentase ini masih dalam batas kriteria kelulusan (maksimal 10 poin), tetapi tetap perlu dijelaskan. Kelas yang paling lemah adalah **Kosong pada data privat, dengan akurasi 87,10%**: ruangan kosong yang direkam dengan ponsel lebih sering salah dikenali dibanding ruangan kosong pada data publik. Penjelasan yang paling mungkin adalah perbedaan kondisi perekaman. Sensor suara pada dataset UCI dipasang tetap di ruangan kantor dengan lingkungan yang relatif terkendali, sedangkan ponsel dalam proyek ini merekam di ruangan biasa yang lebih terbuka terhadap suara dari luar. Akibatnya, ruangan kosong pada data privat lebih sering terdengar "berisik" dan lebih mudah salah diklasifikasikan sebagai Terisi. Temuan ini jujur menunjukkan batas dari pendekatan dua sensor: begitu kondisi lingkungannya berubah, kinerja model ikut berubah.

### 4.4 Hasil di Google Colab

**Intinya:** seluruh langkah — dari unduh data sampai evaluasi — ada dalam satu notebook Colab, dan saat dijalankan menghasilkan angka yang sama dengan yang dilaporkan di sini.

Notebook Google Colab proyek ini berisi langkah-langkah yang sama persis dengan Bab III: mengunduh dataset UCI, memeriksa data, menghitung fitur, menggabungkan data publik dan privat, membagi data, melatih Decision Tree dengan GridSearchCV, lalu mengevaluasinya. Notebook yang sudah dieksekusi menunjukkan hasil yang sama: akurasi uji 95,33%, presisi 95,95%, recall 94,67%, dan F1-score 95,30%, dengan confusion matrix [[144, 6], [8, 142]]. Artinya, hasil laporan ini bisa dicek ulang oleh siapa pun yang menjalankan notebook itu — tidak ada angka yang hanya tertulis di laporan tanpa bisa dibuktikan dari kode.

<!-- SCREENSHOT: docs/screenshots/colab-hasil.png — Hasil eksekusi notebook Colab -->

### 4.5 Hasil Pengujian Web dan Fungsional

**Intinya:** semua halaman bisa dibuka, prediksi dan riwayat berfungsi, rekomendasi yang tampil sesuai dengan hasil prediksinya, dan fitur Tambah Dataset pada demo publik berjalan seperti yang direncanakan.

**Versi lengkap (Flask).** Aplikasi web RoomGuard berhasil dijalankan secara lokal dengan Flask dan SQLite. Seluruh halaman — Dashboard, Predict, History, Dataset, Model, dan About — dapat dibuka tanpa error. Form prediksi menerima nilai Lux dan dB, menampilkan hasil klasifikasi beserta persentase keyakinan, dan menyimpan hasilnya ke riwayat. Rekomendasi hemat energi yang tampil sudah sesuai dengan hasil prediksi: prediksi Kosong menampilkan saran untuk mematikan lampu dan pendingin ruangan, sedangkan prediksi Terisi menampilkan saran pengaturan normal.

Fitur CRUD pada riwayat juga berjalan: satu entri bisa diubah dan dihapus, dan seluruh riwayat bisa dibersihkan. Pengujian input di luar rentang wajar (misalnya Lux negatif atau dB di atas 85) tidak membuat aplikasi crash; sistem menampilkan pesan yang meminta pengguna memeriksa kembali masukannya. Semua kriteria pengujian fungsional dari Bab III terpenuhi.

<!-- SCREENSHOT: docs/screenshots/web-prediksi.png — Tampilan halaman prediksi web RoomGuard -->

**Demo publik (GitHub Pages).** Demo satu halaman ini diuji secara terpisah dengan browser otomatis. Form klasifikasi bekerja: input 180 Lux dan 58 dB menghasilkan "Ruangan Terisi" beserta alasan dan sarannya. Fitur Tambah Dataset diuji langkah demi langkah. Menambahkan satu baris lewat form (150 Lux, 52 dB, Terisi) membuat penghitung berubah dari total 2.000 menjadi 2.001 baris, dan baris itu muncul di tabel. Mengunggah file CSV berisi dua baris valid dan satu baris di luar rentang menghasilkan pesan "2 baris berhasil ditambahkan, 1 baris dilewati", dan totalnya menjadi 2.003 baris. Menghapus satu baris dari tabel menurunkan total menjadi 2.002 baris. Setelah halaman dimuat ulang, data tambahan tetap tersimpan dengan total yang sama, 2.002 baris. Tombol Unduh CSV menghasilkan file `roomguard_data_tambahan.csv` yang isinya sama dengan tabel. Terakhir, memasukkan Cahaya 700 Lux — di luar rentang 0–600 — menampilkan pesan galat "Cahaya harus 0–600 Lux dan suara 25–85 dB", dan penghitung tidak berubah. Selama pengujian ini tidak muncul error JavaScript. Dengan begitu, seluruh kriteria pengujian demo pada Bab III terpenuhi.

<!-- SCREENSHOT: docs/screenshots/web-tambah-dataset.png — Bagian Tambah Dataset pada web demo -->

### 4.6 Pembahasan dan Keterbatasan

**Intinya:** dua sinyal ponsel sudah cukup untuk akurasi 95,33% — tetapi data privatnya masih sempit, suara luar tidak bisa dibedakan, dan web belum terhubung langsung ke sensor.

Hasil proyek ini menunjukkan bahwa dua sinyal sensor ponsel — cahaya dan suara — sudah cukup untuk mengklasifikasikan keberadaan orang dengan akurasi 95,33% pada data uji. Suara menjadi penentu utama, cahaya menjadi pelengkap. Model yang dihasilkan juga mudah dijelaskan, dan aplikasinya sudah bisa dipakai untuk demonstrasi.

Namun ada beberapa keterbatasan yang perlu diakui secara terbuka.

Pertama, **data privat hanya berasal dari satu jenis ruangan dan satu ponsel**. Lima sesi pengambilan data dilakukan di ruangan yang sama dengan perangkat yang sama, sehingga belum diketahui bagaimana model bekerja di ruangan yang lebih besar, lebih bising, atau dengan ponsel yang berbeda.

Kedua, **suara dari luar ruangan tidak bisa dibedakan**. Mikrofon ponsel merekam semua suara, termasuk suara koridor dan ruangan sebelah. Inilah penyebab utama false positive pada data privat, terutama pada kelas Kosong yang akurasinya hanya 87,10%.

Ketiga, **model hanya memakai dua fitur**. Sensor lain seperti suhu, kelembapan, atau CO₂ — yang tersedia di dataset UCI — sengaja tidak dipakai, tetapi penelitian terdahulu menunjukkan sensor-sensor itu bisa membantu, terutama untuk kasus ruangan terisi yang hening.

Keempat, **aplikasi web masih berjalan lokal dan belum terhubung langsung dengan sensor ponsel secara real-time**. Pengguna masih harus memasukkan nilai Lux dan dB secara manual, sehingga sistem belum sepenuhnya otomatis.

Kelima, **data tambahan dari fitur Tambah Dataset belum dipakai untuk melatih ulang model**. Data yang dimasukkan pengunjung hanya tersimpan di browser pengunjung sendiri, sehingga model tidak belajar apa-apa dari data baru itu sampai ada proses pelatihan ulang di Colab.

Keterbatasan-keterbatasan ini bukan kegagalan, melainkan batas yang wajar dari ruang lingkup proyek ini. Semuanya dicatat di sini supaya pembaca dan penguji bisa menilai hasil proyek secara proporsional, dan supaya penelitian lanjutan punya titik awal yang jelas.

---

## BAB V — PENUTUP

### 5.1 Kesimpulan

**Intinya:** modelnya jadi, angkanya bagus, dan web-nya berjalan — dengan catatan jujur pada data privat.

Jadi, dari semua yang dikerjakan dan diuji, ada empat hal yang bisa dibilang:

1. Model Decision Tree berhasil dibangun untuk mengklasifikasikan status keberadaan orang dalam ruangan (Terisi atau Kosong) hanya dari dua fitur sensor ponsel, yaitu Light_Lux dan Sound_dB. Pengaturan terbaik yang ditemukan lewat GridSearchCV adalah kriteria gini, `max_depth` 5, dan `min_samples_leaf` 10, dengan akurasi validasi silang 97,29%.
2. Pada data uji sebanyak 300 baris, model mencapai akurasi 95,33%, presisi 95,95%, recall 94,67%, dan F1-score 95,30%. Dari 300 data, 286 diprediksi dengan benar; kesalahannya terdiri dari 6 false positive dan 8 false negative. Kinerja per sumber data adalah 96,94% pada data publik (n = 229) dan 90,14% pada data privat (n = 71), dengan kelas Kosong pada data privat sebagai yang terlemah (87,10%).
3. Fitur Sound_dB adalah penentu utama keputusan model dengan importance 0,9006, sedangkan Light_Lux berperan sebagai pelengkap dengan importance 0,0994. Aturan utama pohon: suara di bawah atau sama dengan 41,5 dB cenderung Kosong, di atas 46,35 dB cenderung Terisi, dan di antaranya cahaya membantu memutuskan.
4. Aplikasi web RoomGuard berhasil dibangun dalam dua wujud: versi lengkap Flask dan SQLite dengan halaman Dashboard, Predict, History, Dataset, Model, dan About, serta fitur prediksi, rekomendasi hemat energi, dan pengelolaan riwayat; dan demo publik berbahasa Indonesia di GitHub Pages dengan fitur Tambah Dataset, yang memungkinkan pengunjung menambahkan data sendiri lewat form atau unggah CSV. Seluruh fungsi yang direncanakan sudah diuji dan berjalan.

### 5.2 Saran

**Intinya:** tambah variasi ruangan, tambah fitur, sambungkan sensor secara langsung, dan bandingkan dengan model lain.

Untuk pengembangan selanjutnya, ada beberapa hal yang bisa dilakukan:

1. Menambah variasi data privat: mengambil data di beberapa ruangan yang berbeda, dengan ponsel yang berbeda, dan pada jam yang berbeda, supaya model lebih tahan terhadap perubahan kondisi lingkungan.
2. Menambah fitur dari sensor lain, misalnya data akselerometer atau pola perubahan cahaya dari waktu ke waktu, untuk membantu membedakan ruangan terisi yang hening dari ruangan kosong yang berisik.
3. Menghubungkan aplikasi web langsung dengan sensor ponsel secara real-time, misalnya lewat aplikasi Phyphox atau sensor berbasis web, supaya pengguna tidak perlu memasukkan nilai secara manual.
4. Membandingkan Decision Tree dengan model lain, seperti Random Forest, untuk melihat apakah akurasi bisa ditingkatkan tanpa mengorbankan kemudahan interpretasi.
5. Menguji sistem dalam jangka waktu yang lebih panjang, misalnya selama satu minggu penuh, untuk melihat kestabilan kinerjanya pada pola penggunaan ruangan yang berubah-ubah.
6. Menyambungkan data tambahan dari fitur Tambah Dataset ke proses pelatihan ulang, supaya data yang dimasukkan pengunjung benar-benar bisa memperbaiki model, bukan hanya tersimpan di browser.

---

## DAFTAR PUSTAKA

Candanedo, L. M., & Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO₂ measurements using statistical learning models. *Energy and Buildings*, 112, 28–39.

Akkaya, K., Guvenc, I., Aygun, R., Pala, N., & Kadri, A. (2015). IoT-based occupancy monitoring techniques for energy-efficient smart buildings. *IEEE Wireless Communications and Networking Conference Workshops (WCNCW)*, 58–63.

Wang, D., dkk. (2017). Occupancy estimation based on audio signals in indoor environments. *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

Zou, H., dkk. (2018). Light and motion sensor based occupancy detection for smart buildings. *IEEE Sensors Journal*, 18(8), 3365–3374.

UCI Machine Learning Repository. (2024). *Room Occupancy Estimation Dataset (Dataset #864)*. Diakses 24 November 2026 dari https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation

Phyphox. (2026). *Phyphox — Physical Phone Experiments*. RWTH Aachen University. Diakses November 2026 dari https://phyphox.org/

Pedregosa, F., dkk. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Chapman & Hall.

Flask Documentation. (2026). *Flask Web Framework*. Diakses November 2026 dari https://flask.palletsprojects.com/

SQLite Documentation. (2026). *SQLite*. Diakses November 2026 dari https://www.sqlite.org/

---

*Lampiran (notebook Google Colab, kode aplikasi web, dan dataset gabungan) tersedia pada repositori GitHub proyek RoomGuard.*
