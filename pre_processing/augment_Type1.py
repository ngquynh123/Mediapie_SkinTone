import os
import cv2
import numpy as np
from tqdm import tqdm

# ====== Cấu hình ======
INPUT_DIR = "public/data_3/dataset_cheeks_skin/Type_1"     # Thư mục chứa ảnh Type_1 gốc
OUTPUT_DIR = "public/data_3/augmented_dataset/Type_1"      # Thư mục lưu ảnh đã tăng cường
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== Hàm tăng sáng hoặc làm tối ảnh ======
def adjust_brightness(image, factor=1.0):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

# ====== Hàm xoay ảnh nhỏ góc ======
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# ====== Hàm tạo ảnh tăng cường ======
def augment_image(img):
    aug_imgs = []

    # 1. Ảnh gốc
    aug_imgs.append(img)

    # 2. Flip ngang
    aug_imgs.append(cv2.flip(img, 1))

    # 3. Tăng và giảm sáng
    aug_imgs.append(adjust_brightness(img, 1.1))  # sáng hơn
    aug_imgs.append(adjust_brightness(img, 0.9))  # tối hơn

    # 4. Xoay trái/phải nhỏ
    aug_imgs.append(rotate_image(img, 5))
    aug_imgs.append(rotate_image(img, -5))

    # 5. Làm mờ nhẹ
    aug_imgs.append(cv2.GaussianBlur(img, (3, 3), 0))

    return aug_imgs

# ====== Tăng cường tất cả ảnh trong thư mục ======
count = 0
for fname in tqdm(os.listdir(INPUT_DIR), desc="🔄 Đang tăng cường Type_1"):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Lỗi đọc ảnh: {fname}")
        continue

    augmented = augment_image(img)
    base_name = os.path.splitext(fname)[0]

    for idx, aug_img in enumerate(augmented):
        save_name = f"{base_name}_aug{idx}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, aug_img)
        count += 1

print(f"\n✅ Đã tăng cường xong Type_1. Tổng ảnh sau tăng cường: {count}")
