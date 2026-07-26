# NOTE COMVISAASRECOG.PY MERUPAKAN CODE UNTUK DATASET INDONESIAN LICENSE PLATE RECOGNITION
# COMVISAASV2.PY MERUPAKAN CODE UNTUK DATASET INDONESIAN LICENTE PLATE DATASET
# OCR Plat Nomor Kendaraan menggunakan Visual Language Model (LM Studio)

Program OCR plat nomor kendaraan Indonesia menggunakan Visual Language Model
(VLM) yang dijalankan lokal via **LM Studio**, diintegrasikan dengan Python.

- **Model**: SmolVLM2-2.2B-Instruct (via LM Studio Local Server)
- **Dataset**: Indonesian License Plate Dataset (folder `test`, format YOLO)
- **Metrik evaluasi**: Character Error Rate (CER)

---

## Instruksi Eksekusi

### 1. Persiapan Dataset

1. Download dataset dari Kaggle:
   https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset
2. Ekstrak file zip-nya. Letakkan folder hasil ekstrak sejajar dengan
   `ocrv2.py`, sehingga strukturnya seperti ini:

```
plate_ocr/
├── Indonesian License Plate Dataset/
│   ├── images/test/xxx.jpg
│   ├── labels/test/xxx.txt
│   └── labelswithLP/test/xxx.txt   <- dipakai sebagai ground truth
├── ocrv2.py
├── requirements.txt
└── README.md
```

3. Jika lokasi folder dataset berbeda, ubah variabel `DATASET_ROOT` di
   bagian atas `ocrv2.py`:

```python
DATASET_ROOT = "./Indonesian License Plate Dataset"
```

### 2. Jalankan LM Studio

1. Buka aplikasi **LM Studio**.
2. Download & load model **SmolVLM2-2.2B-Instruct** (atau model VLM lain
   yang kompatibel, lihat catatan di bawah).
3. Buka tab **Local Server** (ikon `<->`), pilih model tersebut, klik
   **Start Server**.
4. Pastikan server aktif di `http://127.0.0.1:1234`. Jika port berbeda,
   sesuaikan `LMSTUDIO_URL` di `ocrv2.py`:

```python
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
```

5. Cek nama model yang muncul persis di LM Studio, lalu sesuaikan
   `MODEL_NAME` di `ocrv2.py` jika perlu.

### 3. Setup Environment Python

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 4. Jalankan Program

Pastikan LM Studio Local Server sudah menyala, lalu:

```bash
python ocrv2.py
```

Program akan berjalan otomatis:
1. Membaca seluruh file label di `labelswithLP/test/`.
2. Meng-crop tiap plat dari gambar sesuai bounding box.
3. Mengirim tiap crop plat ke LM Studio untuk dibaca.
4. Menghitung CER dari hasil prediksi vs ground truth.
5. Menyimpan hasil ke `hasil_ocr_platv1.csv`.
6. Menampilkan rata-rata CER di terminal saat selesai.

Contoh output di terminal:

```
[1/100] test001: 3 plat terdeteksi
    [test001_0] GT: 'B9140BCD' | Pred: 'B9140BCD' | CER: 0.0
...
[SELESAI] Total plat diproses: 197
[HASIL] Disimpan di hasil_ocr_platv1.csv
[RATA-RATA CER] 0.1082
```

### 5. Melihat Hasil

Buka `hasil_ocr_platv1.csv`, berisi kolom:

```
image, ground_truth, prediction, CER_score
```

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Connection refused` saat request ke LM Studio | Pastikan Local Server LM Studio sudah di-**Start**, cek port di `LMSTUDIO_URL` |
| `FileNotFoundError` folder dataset | Cek path `DATASET_ROOT` sesuai lokasi folder dataset hasil ekstrak |
| Prediksi kosong / error terus-menerus | Cek nama model di `MODEL_NAME` sudah sesuai dengan yang di-load di LM Studio |
| CER tinggi / prediksi ikut baca tanggal masa berlaku plat | Sudah dimitigasi lewat prompt & regex di `clean_prediction()`, lihat komentar di kode |

## Struktur File

```
plate_ocr/
├── ocrv2.py
├── requirements.txt
├── hasil_ocr_platv1.csv
└── README.md
```

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
