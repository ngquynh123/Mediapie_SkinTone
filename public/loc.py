import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab, deltaE_cie76
import shutil

# === Cấu hình thư mục ===
INPUT_DIR = "D:/KLTN/SKINTONE/public/data_3/dataset_cheeks_skin/Type_3"      # VD: "data/cheeks"
OUTPUT_DIR = "D:/KLTN/SKINTONE/public/data_3/new_tone3"   # VD: "data/filtered_tone3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Giá trị mẫu LAB của tone Type_3 ===
sample_lab_type3 = np.array([89.6501832, -3.77513675, 29.18819513])

# === Ngưỡng tối ưu để nhận Type_3 (có thể chỉnh) ===
THRESHOLD = 12  # giá trị nhỏ hơn nghĩa là gần hơn tone 3

# === Hàm tính trung bình LAB của ảnh ===
def get_average_lab(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = rgb2lab(img_rgb)
    avg_lab = np.mean(img_lab.reshape(-1, 3), axis=0)
    return avg_lab

# === Bắt đầu lọc ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"Tổng ảnh đầu vào: {len(image_files)}")

matched = 0
for filename in tqdm(image_files, desc="🧪 Lọc Type_3"):
    img_path = os.path.join(INPUT_DIR, filename)
    avg_lab = get_average_lab(img_path)
    if avg_lab is None:
        continue

    distance = deltaE_cie76(avg_lab, sample_lab_type3)
    if distance < THRESHOLD:
        matched += 1
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, filename))

print(f"\n✅ Số ảnh khớp tone Type_3: {matched}/{len(image_files)}")
