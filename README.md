# OCR Plat Nomor Kendaraan menggunakan Visual Language Model (LM Studio)

## Deskripsi

Program ini melakukan Optical Character Recognition (OCR) pada plat nomor
kendaraan Indonesia menggunakan Visual Language Model (VLM), yang dijalankan
secara lokal melalui **LM Studio** dan diintegrasikan dengan Python.

- **Model**: SmolVLM2-2.2B-Instruct (via LM Studio Local Server)
- **Dataset**: Indonesian License Plate Dataset (folder `test`, format YOLO)
- **Metrik evaluasi**: Character Error Rate (CER)

## Dataset

Dataset mengikuti struktur YOLO:

```
images/test/xxx.jpg          -> gambar kendaraan
labelswithLP/test/xxx.txt    -> class x_center y_center width height PLATE_TEXT
```

Setiap baris di `labelswithLP` merepresentasikan satu plat (bounding box +
teks plat asli sebagai ground truth). Satu gambar bisa memiliki lebih dari
satu plat.

## Metode

1. Untuk setiap plat pada `labelswithLP/test/`, gambar di-**crop** sesuai
   bounding box-nya.
2. Crop plat dikirim ke LM Studio (format OpenAI-compatible API) dengan
   prompt:
   > "This is a cropped image of an Indonesian vehicle license plate...
   > Respond with ONLY the main plate number."
3. Hasil prediksi dibersihkan dengan regex untuk memisahkan nomor plat
   utama dari teks tambahan (tanggal masa berlaku) yang kadang ikut terbaca.
4. Prediksi dibandingkan dengan ground truth menggunakan **CER**:

```
CER = (S + D + I) / N
```
S = substitusi, D = penghapusan, I = penyisipan, N = jumlah karakter ground truth.
Dihitung dengan algoritma edit distance (Levenshtein) + traceback.

5. Hasil disimpan ke `hasil_ocr_plat.csv` dengan kolom:
   `image, ground_truth, prediction, CER_score`

## Cara Menjalankan

```bash
# 1. Jalankan LM Studio, load model SmolVLM2-2.2B-Instruct, start Local Server
# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan program
python ocr_plate_lmstudio.py
```

Sesuaikan `DATASET_ROOT` di dalam script dengan lokasi folder dataset di
komputer masing-masing.

## Hasil Eksperimen

Diuji pada 100 gambar / 197 plat dari folder `test`:

| Metrik | Nilai |
|---|---|
| Rata-rata CER | **0.1082** |
| Prediksi sempurna (CER = 0) | 133 / 197 (67.5%) |
| CER ≤ 0.2 | 164 / 197 (83.2%) |

Contoh baris hasil:

| image | ground_truth | prediction | CER_score |
|---|---|---|---|
| test008_0 | DK1157AAB | DK1157AAB | 0.0 |
| test008_1 | AA1997FE | AA1997EE | 0.125 |

## Analisis

- Prompt awal yang polos ("Respond only with the plate number") membuat
  model ikut membaca tanggal masa berlaku plat, sehingga rata-rata CER
  awalnya **~0.51**. Setelah prompt diperjelas dan ditambah post-processing
  regex, CER turun menjadi **~0.11** (turun ±79%).
- Kesalahan yang tersisa umumnya berupa kekeliruan karakter mirip
  (`B`↔`8`, `O`↔`0`/`Q`, `G`↔`C`), wajar untuk model VLM kecil (2.2B) yang
  bersifat general-purpose, bukan model OCR khusus plat nomor.

## Keterbatasan

- Model bukan model OCR khusus, sehingga akurasi karakter individual bisa
  keliru pada kondisi pencahayaan/blur tertentu.
- Regex ekstraksi plat mengasumsikan format umum plat Indonesia dan belum
  menutupi seluruh variasi format daerah.

## Diskusi: Problem Menarik yang Ditemukan

Beberapa temuan selama eksperimen yang layak didiskusikan lebih dalam:

1. **VLM ikut "membaca" informasi yang tidak diminta.**
   Plat Indonesia mencetak dua jenis teks berdekatan: nomor plat (besar) dan
   tanggal masa berlaku (kecil, di bawah). Dengan prompt naif, model
   menganggap keduanya sebagai satu string yang harus dibaca semua, karena
   secara visual keduanya sama-sama "teks di gambar". Ini menunjukkan bahwa
   VLM general-purpose tidak otomatis memahami *struktur semantik* objek
   seperti plat nomor — ia butuh instruksi eksplisit tentang bagian mana
   yang relevan. Ini beda dengan OCR tradisional (mis. Tesseract) yang
   memang cuma menangkap teks tanpa menyeleksi maknanya, tapi juga tidak
   bisa membedakan "field mana yang penting" tanpa post-processing manual —
   jadi masalah ini sebenarnya berpindah tempat (dari image-processing ke
   prompt-engineering), bukan hilang.

2. **Prompt engineering memberi dampak lebih besar daripada dugaan awal.**
   Rata-rata CER turun ~79% (dari 0.51 ke 0.11) hanya dengan memperjelas
   instruksi prompt dan menambah post-processing regex, tanpa mengganti
   model maupun mengubah cara crop gambar. Ini menarik karena menunjukkan
   pada model VLM ukuran kecil, kualitas output kadang lebih ditentukan
   oleh cara "bertanya" daripada kapabilitas model itu sendiri.

3. **Regex post-processing punya trade-off.**
   Regex yang dipakai untuk memotong teks tambahan (tanggal validasi)
   berasumsi format plat "huruf-angka-huruf". Ini efektif untuk mayoritas
   kasus, tapi gagal pada plat dengan akhiran huruf+angka (mis. `V0`, `W0`)
   yang dipakai di beberapa daerah — regex yang terlalu ketat malah
   memotong bagian yang seharusnya jadi jawaban benar. Ini contoh nyata
   trade-off klasik di NLP/pattern-matching: aturan yang terlalu spesifik
   berisiko overfit ke pola mayoritas dan gagal pada kasus minoritas.

4. **Kesalahan karakter mengikuti pola visual, bukan acak.**
   Kesalahan yang tersisa hampir selalu antara karakter yang mirip secara
   bentuk (`B`↔`8`, `O`↔`0`/`Q`, `G`↔`C`), bukan kesalahan acak. Ini
   konsisten dengan cara kerja VLM yang "melihat" bentuk visual, mirip
   pola kesalahan OCR klasik — indikasi bahwa error di sini lebih banyak
   disebabkan resolusi/kualitas crop gambar plat, bukan kegagalan
   pemahaman bahasa model.

5. **Crop per-bounding-box vs OCR gambar utuh.**
   Karena satu gambar bisa memiliki lebih dari satu plat, pendekatan
   meng-crop tiap bounding box sebelum OCR terbukti penting: mengirim
   gambar utuh ke model akan membuatnya bingung menentukan plat mana yang
   dimaksud, terutama saat ada 2 kendaraan dalam satu frame. Ini
   menunjukkan bahwa integrasi deteksi objek (bounding box) dengan VLM
   text-reading adalah pola yang lebih andal dibanding mengandalkan VLM
   untuk sekaligus mendeteksi dan membaca teks dari gambar kompleks.

## Kesimpulan

Program berhasil mengintegrasikan Visual Language Model (SmolVLM2-2.2B-Instruct)
yang dijalankan via LM Studio dengan Python untuk melakukan OCR plat nomor
kendaraan, lengkap dengan evaluasi kuantitatif menggunakan CER sesuai rumus
yang ditentukan.

Hasil akhir menunjukkan rata-rata CER **0.1082**, dengan **67.5%** prediksi
yang cocok sempurna dengan ground truth pada 197 plat yang diuji. Pipeline
crop-per-bounding-box + prompt yang eksplisit + post-processing regex
terbukti efektif menurunkan CER hingga ±79% dibanding pendekatan naif.

Meski demikian, hasil ini juga menunjukkan keterbatasan model VLM kecil
(2.2B parameter) untuk tugas OCR presisi tinggi: kesalahan yang tersisa
didominasi oleh kekeliruan karakter visual mirip dan variasi format plat
daerah yang belum sepenuhnya tertangani oleh aturan regex yang digunakan.
Secara keseluruhan, eksperimen ini menegaskan bahwa performa VLM untuk
tugas berformat ketat seperti OCR sangat bergantung pada kombinasi
prompt engineering dan post-processing, bukan hanya pada kapabilitas model
mentahnya.

## Struktur File

```
plate_ocr/
├── ocrv2.py
├── requirements.txt
├── hasil_ocr_platv1.csv
└── README.md
```
