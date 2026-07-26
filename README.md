# OCR Plat Nomor Kendaraan Menggunakan Visual Language Model (LM Studio)

Program ini digunakan untuk melakukan OCR (Optical Character Recognition) pada plat nomor kendaraan Indonesia menggunakan Visual Language Model (VLM) yang dijalankan secara lokal melalui **LM Studio**.

> **Catatan**
>
> - `comvisaasrecog.py` digunakan untuk dataset **Indonesian License Plate Recognition**.
> - `comvisaasv2.py` digunakan untuk dataset **Indonesian License Plate Dataset**.

## Model

- Qwen2-VL-2B-Instruct (melalui LM Studio Local Server)

## Dataset

- Indonesian License Plate Recognition
- Format YOLO (bounding box per karakter)

## Metrik Evaluasi

- Character Error Rate (CER)
- Exact Match Accuracy

---

# Instruksi Eksekusi

## 1. Persiapan Dataset

Download dataset **Indonesian License Plate Recognition**, kemudian letakkan dengan struktur folder berikut:

```
project/
├── Indonesian License Plate Recognition Dataset/
│   ├── images/
│   │   └── test/
│   ├── labels/
│   │   └── test/
│   └── classes.names
├── comvisaasrecog.py
├── comvisaasv2.py
├── requirements.txt
└── README.md
```

Apabila lokasi dataset berbeda, ubah variabel berikut pada program:

```python
DATASET_ROOT = "./Indonesian License Plate Recognition Dataset"
```

---

## 2. Jalankan LM Studio

1. Buka LM Studio.
2. Download dan load model **Qwen2-VL-2B-Instruct**.
3. Masuk ke menu **Local Server**.
4. Jalankan server.
5. Pastikan server aktif pada:

```
http://127.0.0.1:1234
```

Apabila menggunakan port lain, ubah konfigurasi berikut:

```python
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
```

Pastikan nama model sesuai dengan konfigurasi:

```python
MODEL_NAME = "qwen2-vl-2b-instruct"
```

---

## 3. Install Dependensi

```bash
pip install pillow requests
```

atau

```bash
pip install -r requirements.txt
```

---

## 4. Menjalankan Program

Untuk dataset **Indonesian License Plate Recognition**:

```bash
python comvisaasrecog.py
```

Untuk dataset **Indonesian License Plate Dataset**:

```bash
python comvisaasv2.py
```

Program akan secara otomatis:

1. Membaca seluruh gambar dan label YOLO.
2. Membentuk ground truth dari file label.
3. Mengirim gambar ke LM Studio.
4. Mengekstrak hasil OCR plat nomor.
5. Menghitung Character Error Rate (CER).
6. Menghitung Exact Match Accuracy.
7. Menyimpan hasil evaluasi ke file CSV.
8. Menyimpan ringkasan hasil evaluasi.
9. Menyimpan gambar yang dikirim ke model pada folder `debug_crops`.

---

# Output

Program menghasilkan beberapa file berikut.

```
recogcomvis.csv
```

Berisi hasil prediksi setiap gambar.

Kolom:

```
image
ground_truth
prediction
raw_output
CER_score
```

Selain itu akan dihasilkan:

```
recogcomvis_summary.txt
```

yang berisi:

- Total data yang diproses
- Rata-rata CER
- Exact Match Accuracy
- Model yang digunakan
- Dataset split

Folder berikut juga akan dibuat:

```
debug_crops/
```

yang berisi gambar yang dikirim ke model untuk proses debugging.

---

# Konfigurasi

Beberapa parameter yang dapat diubah pada bagian awal program:

| Variabel | Keterangan |
|----------|------------|
| `LMSTUDIO_URL` | Alamat API LM Studio |
| `MODEL_NAME` | Nama model yang digunakan |
| `DATASET_ROOT` | Lokasi dataset |
| `SPLIT` | Dataset split (`test`) |
| `USE_ORIGINAL_IMAGE` | Menggunakan gambar asli atau hasil crop |
| `ENABLE_ENHANCE` | Mengaktifkan image enhancement |
| `CROP_PADDING_RATIO` | Padding saat crop |
| `MAX_IMAGE_SIZE` | Resolusi maksimum gambar |
| `JPEG_QUALITY` | Kualitas kompresi JPEG |

---

# Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Tidak dapat terhubung ke LM Studio | Pastikan Local Server telah dijalankan dan URL sesuai. |
| Model tidak ditemukan | Pastikan nama model sama dengan yang dimuat di LM Studio. |
| Dataset tidak ditemukan | Periksa kembali lokasi `DATASET_ROOT`. |
| File `classes.names` tidak ditemukan | Pastikan file tersedia pada folder dataset. |
| Hasil OCR kosong | Pastikan gambar jelas dan model berhasil dimuat. |

---

# Struktur File

```
project/
├── comvisaasrecog.py
├── comvisaasv2.py
├── requirements.txt
├── README.md
├── recogcomvis.csv
├── recogcomvis_summary.txt
└── debug_crops/
```

---

# Kesimpulan

Program ini mengimplementasikan OCR plat nomor kendaraan Indonesia menggunakan model **Qwen2-VL-2B-Instruct** yang dijalankan melalui **LM Studio**. Sistem melakukan pembacaan dataset berformat YOLO, mengekstrak teks plat nomor menggunakan Visual Language Model, kemudian mengevaluasi hasil prediksi menggunakan **Character Error Rate (CER)** dan **Exact Match Accuracy**. Seluruh hasil prediksi, ringkasan evaluasi, dan gambar debugging disimpan secara otomatis sehingga memudahkan proses analisis performa model.
