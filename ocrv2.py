
import os
import re
import csv
import glob
import base64
import time
import io
import requests
from PIL import Image


# KONFIGURASI - SESUAIKAN DENGAN SETUP KAMU

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "smolvlm2-2.2b-instruct"

# Root folder dataset (hasil ekstrak zip Kaggle)
DATASET_ROOT = "./Indonesian License Plate Dataset"

IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "test")
LABELS_LP_DIR = os.path.join(DATASET_ROOT, "labelswithLP", "test")

OUTPUT_CSV = "hasil_ocr_platv1.csv"

# Padding tambahan di sekitar bbox crop (persentase dari lebar/tinggi bbox),
# supaya karakter di tepi plat tidak terpotong.
CROP_PADDING_RATIO = 0.10

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PROMPT = (
    "This is a cropped image of an Indonesian vehicle license plate. "
    "Indonesian plates show the main plate number in large characters "
    "(format: 1-2 letters, then digits, then 1-3 letters), and separately "
    "show a small validity/expiry date (month and year) in smaller text, "
    "usually at the bottom. Ignore the small expiry date completely. "
    "Respond with ONLY the main plate number, nothing else, no extra "
    "digits, no explanation."
)


# PARSING LABEL (labelswithLP)


def find_image_path(images_dir: str, stem: str):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
        p_upper = os.path.join(images_dir, stem + ext.upper())
        if os.path.isfile(p_upper):
            return p_upper
    return None


def parse_label_line(line: str):
    """
    Format: class_id x_center y_center width height PLATE_TEXT
    PLATE_TEXT bisa mengandung spasi, jadi ambil 5 token pertama sebagai
    angka, sisanya (token ke-6 dst, digabung) sebagai teks plat.
    """
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    try:
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:5])
    except ValueError:
        return None
    plate_text = " ".join(parts[5:]).strip().upper()
    return {
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "ground_truth": plate_text,
    }



# CROP BOUNDING BOX


def crop_plate(image: Image.Image, box: dict) -> Image.Image:
    img_w, img_h = image.size
    xc, yc, w, h = box["x_center"], box["y_center"], box["width"], box["height"]

    w_pad = w * (1 + CROP_PADDING_RATIO)
    h_pad = h * (1 + CROP_PADDING_RATIO)

    xmin = (xc - w_pad / 2) * img_w
    xmax = (xc + w_pad / 2) * img_w
    ymin = (yc - h_pad / 2) * img_h
    ymax = (yc + h_pad / 2) * img_h

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(img_w, int(xmax))
    ymax = min(img_h, int(ymax))

    return image.crop((xmin, ymin, xmax, ymax))


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# PANGGIL LM STUDIO


def query_lmstudio(crop_image: Image.Image, retries: int = 2) -> str:
    b64_img = image_to_base64(crop_image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LMSTUDIO_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return clean_prediction(text)
        except Exception as e:
            last_err = e
            time.sleep(1)
    print(f"[WARNING] Gagal query LM Studio: {last_err}")
    return ""


def clean_prediction(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return extract_main_plate_pattern(text)


# Pola plat Indonesia: 1-2 huruf, spasi opsional, 1-4 angka, spasi opsional,
# 1-3 huruf. Dipakai untuk memotong teks tambahan (mis. tanggal masa berlaku
# yang kadang ikut terbaca model) di akhir prediksi.
PLATE_PATTERN = re.compile(r"\b([A-Z]{1,2})\s?(\d{1,4})\s?([A-Z]{1,3})\b")


def extract_main_plate_pattern(text: str) -> str:
    match = PLATE_PATTERN.search(text)
    if match:
        letters1, digits, letters2 = match.groups()
        return f"{letters1}{digits}{letters2}"
    # kalau pola tidak cocok sama sekali, kembalikan teks asli (tanpa spasi)
    # supaya tetap bisa dihitung CER-nya (biasanya akan menghasilkan CER tinggi,
    # menandakan prediksi memang gagal total)
    return text.replace(" ", "")



# CHARACTER ERROR RATE (CER)


def levenshtein_distance(ref: str, hyp: str):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    op[0][0] = None

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                if best == sub:
                    op[i][j] = "S"
                elif best == dele:
                    op[i][j] = "D"
                else:
                    op[i][j] = "I"

    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "M":
            i -= 1
            j -= 1
        elif o == "S":
            S += 1
            i -= 1
            j -= 1
        elif o == "D":
            D += 1
            i -= 1
        elif o == "I":
            I += 1
            j -= 1
        else:
            break

    return dp[n][m], S, D, I


def compute_cer(ground_truth: str, prediction: str) -> float:
    ref = ground_truth.replace(" ", "")
    hyp = prediction.replace(" ", "")
    n = len(ref)
    if n == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    _, S, D, I = levenshtein_distance(ref, hyp)
    return round((S + D + I) / n, 4)



def main():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {IMAGES_DIR}")
    if not os.path.isdir(LABELS_LP_DIR):
        raise FileNotFoundError(f"Folder label (labelswithLP) tidak ditemukan: {LABELS_LP_DIR}")

    label_files = sorted(glob.glob(os.path.join(LABELS_LP_DIR, "*.txt")))
    if not label_files:
        raise FileNotFoundError(f"Tidak ada file label di {LABELS_LP_DIR}")

    print(f"[INFO] Ditemukan {len(label_files)} file label di {LABELS_LP_DIR}")

    rows = []
    total_cer = 0.0
    total_plates = 0

    for lf_idx, label_path in enumerate(label_files, 1):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        image_path = find_image_path(IMAGES_DIR, stem)

        if image_path is None:
            print(f"[WARNING] Gambar untuk label {stem} tidak ditemukan, dilewati.")
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l for l in f.readlines() if l.strip()]

        boxes = [parse_label_line(l) for l in lines]
        boxes = [b for b in boxes if b is not None]

        if not boxes:
            continue

        image = Image.open(image_path).convert("RGB")

        print(f"[{lf_idx}/{len(label_files)}] {stem}: {len(boxes)} plat terdeteksi")

        for plate_idx, box in enumerate(boxes):
            ground_truth = box["ground_truth"]
            crop = crop_plate(image, box)
            prediction = query_lmstudio(crop)
            cer_score = compute_cer(ground_truth, prediction)

            total_cer += cer_score
            total_plates += 1

            image_label = f"{stem}_{plate_idx}"
            print(f"    [{image_label}] GT: '{ground_truth}' | Pred: '{prediction}' | CER: {cer_score}")

            rows.append({
                "image": image_label,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "CER_score": cer_score,
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction", "CER_score"])
        writer.writeheader()
        writer.writerows(rows)

    avg_cer = round(total_cer / total_plates, 4) if total_plates else 0.0
    print(f"\n[SELESAI] Total plat diproses: {total_plates}")
    print(f"[HASIL] Disimpan di {OUTPUT_CSV}")
    print(f"[RATA-RATA CER] {avg_cer}")


if __name__ == "__main__":
    main()