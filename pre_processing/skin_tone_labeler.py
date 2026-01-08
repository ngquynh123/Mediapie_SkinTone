import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab
import time

# ========== Cấu hình ==========
RAW_INPUT_PARENT = "D:/KLTN/SKINTONE/pre_processing/combined_new"
TONE_SAMPLE_DIR = "public/skin tone values"
OUTPUT_DIR = "public/data_3/dataset_cheeks_skin"
TEMP_OUTPUT_DIR = "public/data_3/temp_dataset_skintone_tones_cheeks"
ERROR_LOG_PATH = "error_log.txt"

MAX_IMAGES_PER_CLASS = 10000
TONE_GROUPS = ["light", "mid-light", "mid-dark", "dark"]
EXPECTED_TONES = [f"Type_{i}" for i in range(1, 7)]

# ========== Tiện ích ==========
def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

def get_average_color_lab(image_path, resize_dim=(100, 100)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể đọc ảnh")
    img = cv2.resize(img, resize_dim)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    lab_img = rgb2lab(img_rgb).reshape(-1, 3)
    return np.mean(lab_img, axis=0)

# ========== Tính màu trung bình mẫu tone ==========
def compute_tone_references_lab(tone_sample_dir):
    tone_refs = {}
    for folder in os.listdir(tone_sample_dir):
        folder_path = os.path.join(tone_sample_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        tone_number = None
        if folder.isdigit():
            tone_number = int(folder)
        elif folder.lower().startswith("type_") and folder[5:].isdigit():
            tone_number = int(folder[5:])

        if tone_number not in range(1, 7):
            print(f"Bỏ qua thư mục không hợp lệ: {folder}")
            continue

        label = f"Type_{tone_number}"
        tone_colors = []

        for img_name in os.listdir(folder_path):
            if is_image_file(img_name):
                try:
                    color = get_average_color_lab(os.path.join(folder_path, img_name))
                    tone_colors.append(color)
                except Exception as e:
                    print(f"[Lỗi mẫu tông da] {img_name}: {e}")
        if tone_colors:
            tone_refs[label] = np.mean(tone_colors, axis=0)
            print(f" Màu trung bình {label}: {tone_refs[label]}")

    if len(tone_refs) != 6:
        raise ValueError(f"Không đủ 6 mẫu tone da! Có: {list(tone_refs.keys())}")
    return tone_refs

# ========== Gán nhãn tone cho ảnh ==========
def assign_tone_label_lab(image_path, tone_refs):
    try:
        img_color = get_average_color_lab(image_path)
    except:
        raise ValueError("Không thể tính màu LAB")

    l, a, b = img_color

    # ==== Gán nhãn theo điều kiện LAB ====
    # Type_1: Da rất sáng – mở rộng nhẹ
    if 85 <= l <= 100 and -5 <= a <= 6 and -3 <= b <= 14:
        return "Type_1"

    # Type_2: Da sáng – SIẾT CHẶT
    if 87.0 <= l <= 88.2 and 2.0 <= a <= 3.0 and 7.0 <= b <= 10.5:
        return "Type_2"

    # Type_3: Da trung bình sáng – mở rộng vừa
    if 60 <= l <= 88 and -3 <= a <= 15 and 9 <= b <= 45:
        return "Type_3"

    # Type_4: Da trung bình tối – SIẾT CHẶT
    if 55.3 <= l <= 55.8 and 6.6 <= a <= 6.9 and 27.9 <= b <= 28.3:
        return "Type_4"

    # Type_5: Da tối – giữ nguyên
    if 42.5 <= l <= 43.0 and 13.2 <= a <= 13.5 and 23.5 <= b <= 24.5:
        return "Type_5"

    # Type_6: Da rất tối – mở rộng
    if l < 43 and 2 <= a <= 18 and 4 <= b <= 35:
        return "Type_6"

    # ==== Nếu không khớp điều kiện → dùng khoảng cách LAB làm dự phòng ====
    print(f"[Debug] Ảnh {image_path} không khớp điều kiện, dùng cơ chế dự phòng.")
    distances = {label: np.linalg.norm(img_color - ref) for label, ref in tone_refs.items()}
    return min(distances, key=distances.get)

# ========== Gán nhãn tất cả ảnh ==========
def label_all_cheek_images(input_path, output_path, tone_refs, error_log):
    if not os.path.exists(input_path):
        print(f" Không tìm thấy thư mục: {input_path}")
        return

    image_list = [f for f in os.listdir(input_path) if is_image_file(f)]
    for img_name in tqdm(image_list, desc=f" {os.path.basename(input_path)}"):
        img_path = os.path.join(input_path, img_name)
        try:
            label = assign_tone_label_lab(img_path, tone_refs)
            save_folder = os.path.join(output_path, label)
            os.makedirs(save_folder, exist_ok=True)
            shutil.copy(img_path, os.path.join(save_folder, img_name))
        except Exception as e:
            error_log.append(f"{img_path}: {e}")

# ========== Hàm chính ==========
def label_and_copy_images():
    tone_refs = compute_tone_references_lab(TONE_SAMPLE_DIR)
    if not tone_refs:
        return

    start_time = time.time()
    error_log = []

    if os.path.exists(TEMP_OUTPUT_DIR):
        shutil.rmtree(TEMP_OUTPUT_DIR)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    for tone_group in TONE_GROUPS:
        group_path = os.path.join(RAW_INPUT_PARENT, tone_group)
        if os.path.exists(group_path):
            print(f"\n🔄 Xử lý nhóm: {tone_group.upper()}")
            label_all_cheek_images(group_path, TEMP_OUTPUT_DIR, tone_refs, error_log)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for tone in EXPECTED_TONES:
        src = os.path.join(TEMP_OUTPUT_DIR, tone)
        dst = os.path.join(OUTPUT_DIR, tone)
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            for img in os.listdir(src)[:MAX_IMAGES_PER_CLASS]:
                shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    if error_log:
        with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(error_log))
        print(f"⚠️ Có {len(error_log)} lỗi. Đã ghi vào {ERROR_LOG_PATH}")

    end_time = time.time()
    print(f"\n Xong! Thời gian: {end_time - start_time:.2f} giây")

# ========== Chạy ==========
if __name__ == "__main__":
    label_and_copy_images()