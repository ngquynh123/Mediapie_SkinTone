import os
import shutil
import random
from tqdm import tqdm

# ==== CẤU HÌNH ====
INPUT_DIR = "D:/KLTN/SKINTONE/public/data_3/final_data"  # Thư mục chứa Type_1 đến Type_6
OUTPUT_DIR = "D:/KLTN/SKINTONE/public/data_3/final_dataset"        # Kết quả chia train/val/test
SPLITS = ['train', 'val', 'test']

# Tỷ lệ chia
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.1

# === Tạo thư mục output ===
for split in SPLITS:
    for tone in os.listdir(INPUT_DIR):
        split_dir = os.path.join(OUTPUT_DIR, split, tone)
        os.makedirs(split_dir, exist_ok=True)

# === Chia dữ liệu ===
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
for tone in os.listdir(INPUT_DIR):
    tone_dir = os.path.join(INPUT_DIR, tone)
    if not os.path.isdir(tone_dir):
        continue

    image_files = [f for f in os.listdir(tone_dir) if f.lower().endswith(IMAGE_EXT)]
    random.shuffle(image_files)
    total = len(image_files)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    split_data = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }

    for split, files in split_data.items():
        for fname in tqdm(files, desc=f"{tone} → {split}"):
            src = os.path.join(tone_dir, fname)
            dst = os.path.join(OUTPUT_DIR, split, tone, fname)
            shutil.copy(src, dst)

print(" Đã chia dữ liệu xong theo train/val/test.")
