import os
import cv2
import numpy as np
import pandas as pd

# ==== Hàm tính trung bình màu Lab ====
def get_average_lab(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    return np.mean(L), np.mean(a), np.mean(b)

# ==== Xử lý vùng má: dark/light/... → left/right ====
def process_cheek_folder(cheek_root_folder):
    data = []

    for tone_group in os.listdir(cheek_root_folder):  # dark, light, mid-light, ...
        group_path = os.path.join(cheek_root_folder, tone_group)
        if not os.path.isdir(group_path):
            continue

        for side in ['left', 'right']:
            side_path = os.path.join(group_path, side)
            if not os.path.isdir(side_path):
                continue

            for fname in os.listdir(side_path):
                if fname.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(side_path, fname)
                    img = cv2.imread(img_path)
                    if img is not None:
                        L, a, b = get_average_lab(img)
                        data.append({
                            "image": fname,
                            "region": "cheek",
                            "side": side,
                            "group": tone_group,
                            "L": round(L, 2),
                            "a": round(a, 2),
                            "b": round(b, 2),
                            "path": img_path
                        })

    return data

# ==== Xử lý vùng cằm: dark/light/... không có trái/phải ====
def process_chin_folder(chin_root_folder):
    data = []

    for tone_group in os.listdir(chin_root_folder):
        group_path = os.path.join(chin_root_folder, tone_group)
        if not os.path.isdir(group_path):
            continue

        for fname in os.listdir(group_path):
            if fname.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(group_path, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    L, a, b = get_average_lab(img)
                    data.append({
                        "image": fname,
                        "region": "chin",
                        "side": None,
                        "group": tone_group,
                        "L": round(L, 2),
                        "a": round(a, 2),
                        "b": round(b, 2),
                        "path": img_path
                    })

    return data

# ==== Gọi xử lý và lưu file CSV ====
def process_all_and_save(cheek_root, chin_root, output_csv="lab_cheek_chin_data.csv"):
    cheek_data = process_cheek_folder(cheek_root)
    chin_data  = process_chin_folder(chin_root)

    df = pd.DataFrame(cheek_data + chin_data)
    df.to_csv(output_csv, index=False)
    print(f"Đã lưu kết quả vào {output_csv}")
    print(df.head())
    return df

# ==== Ví dụ sử dụng ====
if __name__ == "__main__":
    # Thay đổi đường dẫn nếu cần
    cheek_root = "../pre_processing/data4/output_crop_cheek"
    chin_root  = "../pre_processing/data4/output_crop_chin"
    
    df_lab = process_all_and_save(cheek_root, chin_root)
