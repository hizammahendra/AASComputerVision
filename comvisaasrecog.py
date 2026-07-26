import os
import re
import csv
import glob
import base64
import time
import io
import sys
import requests
from PIL import Image, ImageEnhance, ImageOps

# =============================================================
# KONFIGURASI
# =============================================================
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen2-vl-2b-instruct"

DATASET_ROOT = "./Indonesian License Plate Recognition Dataset"
SPLIT = "test"

IMAGES_DIR = os.path.join(DATASET_ROOT, "images", SPLIT)
LABELS_DIR = os.path.join(DATASET_ROOT, "labels", SPLIT)
CLASSES_FILE = os.path.join(DATASET_ROOT, "classes.names")

OUTPUT_CSV = "recogcomvis.csv"
OUTPUT_SUMMARY = "recogcomvis_summary.txt"
DEBUG_DIR = "debug_crops"                 # folder untuk menyimpan gambar yang dikirim ke model

# Atur ulang perilaku crop & enhancement
USE_ORIGINAL_IMAGE = True                 # langsung pakai gambar asli (karena sudah crop plat)
ENABLE_ENHANCE = False                    # matikan contrast/sharpness sementara
CROP_PADDING_RATIO = 0.0                  # tanpa padding tambahan jika crop manual digunakan

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_IMAGE_SIZE = (1024, 1024)             # resolusi lebih tinggi
JPEG_QUALITY = 85

# Prompt yang sangat ketat + stop token
PROMPT = (
    "Read the Indonesian license plate from this image. "
    "The plate has letters and numbers. "
    "Respond ONLY with the alphanumeric characters, NO spaces, NO punctuation, NO extra text. "
    "Example: if plate is 'B 1234 XYZ', respond 'B1234XYZ'. "
    "If you can't read, respond with an empty string."
)

FALLBACK_PROMPT = "License plate number (letters and digits only):"

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = 120

# ---------------------------- Koneksi ----------------------------
def check_lmstudio_connection():
    base_url = LMSTUDIO_URL.replace("/v1/chat/completions", "/v1/models")
    try:
        resp = requests.get(base_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        available = [m.get("id", "") for m in data.get("data", [])]
        print(f"[INFO] Model tersedia: {available}")
        return MODEL_NAME in available
    except Exception as e:
        print(f"[ERROR] Koneksi gagal: {e}")
        return False

# ---------------------------- Load classes ----------------------------
def load_classes(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]

# ---------------------------- Pencarian gambar ----------------------------
def find_image_path(images_dir, stem):
    for ext in IMG_EXTS:
        for variant in (ext, ext.upper()):
            p = os.path.join(images_dir, stem + variant)
            if os.path.isfile(p):
                return p
    return None

# ---------------------------- Parsing label ----------------------------
def parse_char_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cid = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    return {"class_id": cid, "x_center": x, "y_center": y, "width": w, "height": h}

def build_ground_truth(boxes, classes):
    sorted_boxes = sorted(boxes, key=lambda b: b["x_center"])
    chars = []
    for b in sorted_boxes:
        if 0 <= b["class_id"] < len(classes):
            chars.append(classes[b["class_id"]])
        else:
            print(f"[WARNING] class_id {b['class_id']} out of range")
    return "".join(chars)

# ---------------------------- Crop (hanya jika USE_ORIGINAL_IMAGE=False) ----------------------------
def crop_plate_from_chars(image, boxes):
    img_w, img_h = image.size
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for b in boxes:
        xc, yc, w, h = b["x_center"], b["y_center"], b["width"], b["height"]
        xmins.append(xc - w/2)
        xmaxs.append(xc + w/2)
        ymins.append(yc - h/2)
        ymaxs.append(yc + h/2)
    xmin = max(0, int((min(xmins) - CROP_PADDING_RATIO * (max(xmaxs)-min(xmins))/2) * img_w))
    ymin = max(0, int((min(ymins) - CROP_PADDING_RATIO * (max(ymaxs)-min(ymins))/2) * img_h))
    xmax = min(img_w, int((max(xmaxs) + CROP_PADDING_RATIO * (max(xmaxs)-min(xmins))/2) * img_w))
    ymax = min(img_h, int((max(ymaxs) + CROP_PADDING_RATIO * (max(ymaxs)-min(ymins))/2) * img_h))
    if xmax <= xmin or ymax <= ymin:
        return image
    return image.crop((xmin, ymin, xmax, ymax))

# ---------------------------- Kirim ke LM Studio ----------------------------
def query_lmstudio(image, prompt, stop_tokens=None):
    b64 = image_to_base64(image)
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 30,
        "stop": stop_tokens
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(LMSTUDIO_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            print(f"    [RAW] {raw}")
            return raw
        except Exception as e:
            print(f"[WARNING] Attempt {attempt+1} gagal: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    return ""

def image_to_base64(image):
    img = image.copy()
    img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------------------- Ekstraksi plat ----------------------------
def extract_plate_number(text):
    # Hapus semua non-alfanumerik, uppercase
    raw = re.sub(r'[^A-Z0-9]', '', text.upper())
    if not raw:
        return ""
    # Cari semua kemunculan pola: 1-2 huruf, 1-4 digit, 0-3 huruf
    pattern = re.compile(r'[A-Z]{1,2}\d{1,4}[A-Z]{0,3}')
    matches = pattern.findall(raw)
    if matches:
        return matches[0]  # ambil kemunculan pertama
    # Koreksi pertukaran karakter
    confusion = {'8':'B','B':'8','0':'O','O':'0','1':'I','I':'1','5':'S','S':'5','2':'Z','Z':'2'}
    if 4 <= len(raw) <= 9:
        for i, ch in enumerate(raw):
            if ch in confusion:
                swapped = list(raw)
                swapped[i] = confusion[ch]
                cand = ''.join(swapped)
                m = pattern.search(cand)
                if m:
                    return m.group(0)
    return ""

# ---------------------------- Main ----------------------------
def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Folder gambar tidak ada: {IMAGES_DIR}")
    if not os.path.isdir(LABELS_DIR):
        raise FileNotFoundError(f"Folder label tidak ada: {LABELS_DIR}")

    classes = load_classes(CLASSES_FILE)
    print(f"[INFO] {len(classes)} kelas karakter dimuat")

    if not check_lmstudio_connection():
        print("[ERROR] Model tidak tersedia. Keluar.")
        sys.exit(1)

    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    print(f"[INFO] {len(label_files)} file label ditemukan")

    total_cer = 0.0
    total_plates = 0
    exact_match = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "ground_truth", "prediction", "raw_output", "CER_score"])
        writer.writeheader()

        for idx, label_path in enumerate(label_files, 1):
            stem = os.path.splitext(os.path.basename(label_path))[0]
            img_path = find_image_path(IMAGES_DIR, stem)
            if not img_path:
                print(f"[WARNING] Gambar tidak ditemukan: {stem}")
                continue

            # Baca label
            with open(label_path, "r") as f:
                lines = [l for l in f.readlines() if l.strip()]
            boxes = [parse_char_label_line(l) for l in lines]
            boxes = [b for b in boxes if b is not None]
            if not boxes:
                print(f"[WARNING] Tidak ada bounding box valid: {stem}")
                continue

            gt = build_ground_truth(boxes, classes)
            if not gt:
                print(f"[WARNING] GT kosong: {stem}")
                continue

            image = Image.open(img_path).convert("RGB")
            print(f"\n[{idx}/{len(label_files)}] {stem} | GT: {gt}")

            # Pilih gambar yang akan dikirim
            if USE_ORIGINAL_IMAGE:
                send_img = image.copy()
            else:
                send_img = crop_plate_from_chars(image, boxes)

            # Enhancement (opsional)
            if ENABLE_ENHANCE:
                send_img = ImageEnhance.Contrast(send_img).enhance(1.3)
                send_img = ImageEnhance.Sharpness(send_img).enhance(1.5)
                send_img = ImageOps.autocontrast(send_img, cutoff=2)

            # Simpan debug crop
            debug_path = os.path.join(DEBUG_DIR, f"{stem}_sent.jpg")
            send_img.save(debug_path)

            # Query dengan prompt utama + stop token (berhenti di newline/titik)
            raw1 = query_lmstudio(send_img, PROMPT, stop_tokens=["\n", "."])
            pred = extract_plate_number(raw1)

            # Jika hasil kosong, coba fallback prompt
            if not pred:
                raw2 = query_lmstudio(send_img, FALLBACK_PROMPT, stop_tokens=["\n"])
                pred = extract_plate_number(raw2)
                raw_output = raw2 if raw2 else raw1
            else:
                raw_output = raw1

            cer = compute_cer(gt, pred) if pred else 1.0
            total_cer += cer
            total_plates += 1
            if pred == gt.replace(" ", ""):
                exact_match += 1

            print(f"    Pred: '{pred}' | CER: {cer}")
            writer.writerow({
                "image": stem,
                "ground_truth": gt,
                "prediction": pred,
                "raw_output": raw_output,
                "CER_score": cer
            })

    avg_cer = round(total_cer / total_plates, 4) if total_plates else 0.0
    accuracy = round(exact_match / total_plates, 4) if total_plates else 0.0

    summary = [
        f"Total plat diproses : {total_plates}",
        f"Rata-rata CER       : {avg_cer}",
        f"Exact match         : {exact_match}/{total_plates}",
        f"Accuracy (exact)    : {accuracy}",
        f"Model               : {MODEL_NAME}",
        f"Split               : {SPLIT}",
    ]
    print("\n[SELESAI]")
    for line in summary:
        print(f"[HASIL] {line}")
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print(f"[INFO] CSV -> {OUTPUT_CSV}")
    print(f"[INFO] Ringkasan -> {OUTPUT_SUMMARY}")
    print(f"[INFO] Gambar debug disimpan di folder '{DEBUG_DIR}'")

def compute_cer(ref, hyp):
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein distance
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return round(dp[n][m] / len(ref), 4)

if __name__ == "__main__":
    main()