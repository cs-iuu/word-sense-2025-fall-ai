# =====================================================
# Synthetic OCR dataset generator (YOLOv8 compatible)
# 1 image -> 2~3 text, 1 text -> aug -> 1~3 image
# All labels preserved, no overwrite
# Har irmeg crop + bbox unchanged
# =====================================================

import os
import csv
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A

# ===================== SETTINGS =====================
IMAGE_DIR = "images"
OUT_IMG_DIR = "output/images"
OUT_LBL_DIR = "output/labels"
CSV_FILE = "labels.csv"

FONT_PATH = "arial.ttf"   # Кирилл дэмждэг font
FONT_SIZE = 20
TEXT_COLOR = (255, 0, 0)

MAX_TEXT_PER_IMAGE = 8    # 1 зураг дээр хэдэн текст зэрэг бичих
AUG_PER_TEXT = 3          # 1 текстээс хэдэн augmentation

TEXT_LIST = [
    "Сайн байна уу",
    "Танд баярлалаа",
    "Өнөөдөр сайхан өдөр",
    "Амжилт хүсье",
    "Анхаарна уу",
    "Дахин оролдоно уу",
    "Оруулах боломжгүй",
    "Мэдээлэл буруу байна",
    "Систем ачаалж байна",
    "Түр хүлээнэ үү",
    "Нэвтрэх боломжгүй",
    "Алдаа гарлаа",
    "Мэдээлэл илгээгдлээ",
    "Шинэчлэлт хийгдэж байна",
    "Баталгаажуулна уу",
    "Тохиргоо хадгалагдлаа",
    "Хүсэлт амжилттай",
    "МОНГОЛ",
    "КИРИЛЛ",
    "БИЧИГ",
    "ХИЙМЭЛ ОЮУН",
    "ӨГӨГДӨЛ",
    "СУРАЛЦАХ",
    "ХАРАА",
]

# ===================== FOLDERS =====================
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# ===================== AUGMENT =====================
transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Affine(
            scale=(0.95, 1.05),
            shear=(-4, 4),
            translate_percent=(-0.02, 0.02),
            rotate=(-2, 2),
            p=0.6
        ),
        A.RandomCropFromBorders(
            crop_left=0.02,
            crop_right=0.02,
            crop_top=0.02,
            crop_bottom=0.02,
            p=0.3
        ),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_visibility=0.3,
        clip=True
    )
)

# ===================== MAIN =====================
img_id = 1

with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "text", "x", "y", "w", "h"])

    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        base_img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")

        # 1 зураг дээр санамсаргүй 2~3 текст
        texts = random.sample(TEXT_LIST, k=MAX_TEXT_PER_IMAGE)

        for text in texts:
            img = base_img.copy()
            draw = ImageDraw.Draw(img)

            # Random position
            x = random.randint(20, img.width // 2)
            y = random.randint(20, img.height // 2)

            x1, y1, x2, y2 = draw.textbbox((x, y), text, font=font)
            draw.text((x, y), text, fill=TEXT_COLOR, font=font)

            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            img_w, img_h = img.size
            out_img = f"img_{img_id}.jpg"
            img.save(os.path.join(OUT_IMG_DIR, out_img))

            # CSV
            writer.writerow([out_img, text, x1, y1, bw, bh])

            # YOLO
            xc = (x1 + bw / 2) / img_w
            yc = (y1 + bh / 2) / img_h
            w = bw / img_w
            h = bh / img_h

            lbl_file = os.path.join(OUT_LBL_DIR, out_img.replace(".jpg", ".txt"))
            with open(lbl_file, "a") as lf:
                lf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            img_id += 1

            # ================= AUGMENTATION + HAR IRMEG CROP =================
            bboxes = [[x1, y1, x2, y2]]
            for _ in range(AUG_PER_TEXT):
                aug = transform(
                    image=np.array(img),
                    bboxes=bboxes,
                    category_ids=[0]
                )

                if len(aug["bboxes"]) == 0:
                    continue

                ax1, ay1, ax2, ay2 = map(int, aug["bboxes"][0])
                bw, bh = ax2 - ax1, ay2 - ay1
                if bw <= 0 or bh <= 0:
                    continue

                aug_img = Image.fromarray(aug["image"])
                img_w, img_h = aug_img.size

                # 🔹 Хар ирмэг crop (text bbox үлдэнэ)
                aug_gray = aug_img.convert("L")
                crop_box = aug_gray.getbbox()  # non-background пикселүүдийн хүрээ
                if crop_box:
                    aug_img = aug_img.crop(crop_box)

                out_img_name = f"img_{img_id}.jpg"
                aug_img.save(os.path.join(OUT_IMG_DIR, out_img_name))

                # CSV болон YOLO (bbox өөрчлөгдөхгүй)
                writer.writerow([out_img_name, text, ax1, ay1, bw, bh])
                lbl_file = os.path.join(OUT_LBL_DIR, out_img_name.replace(".jpg", ".txt"))
                with open(lbl_file, "a") as lf:
                    xc = (ax1 + bw / 2) / img_w
                    yc = (ay1 + bh / 2) / img_h
                    w = bw / img_w
                    h = bh / img_h
                    lf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                img_id += 1

print("✅ DONE: 1 image -> 2~3 texts, augmented + har irmeg crop | YOLOv8 READY")
