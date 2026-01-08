import cv2
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== Khởi tạo Mediapipe FaceMesh =====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# ===== Landmark cho vùng cằm =====
chin_indices = [204, 149, 148, 152, 377, 400, 378, 424, 406, 18, 182, 204]
# ===== Landmark má trái & phải =====
LEFT_CHEEK_IDS =  [234, 93,  132, 58,  172, 136, 142]
RIGHT_CHEEK_IDS = [454, 323, 361, 288, 397, 365, 371]

# ===== Hàm kiểm tra landmark hợp lệ =====
def landmarks_valid(landmarks, ids, img_shape):
    h, w = img_shape[:2]
    for i in ids:
        x = landmarks[i].x
        y = landmarks[i].y
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False
        px = int(x * w)
        py = int(y * h)
        if not (0 <= px < w and 0 <= py < h):
            return False
    return True

# ===== Hàm vẽ polygon =====
def draw_polygon(image, landmark_list, results, color):
    h, w = image.shape[:2]
    points = []
    for idx in landmark_list:
        lm = results.multi_face_landmarks[0].landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
    cv2.polylines(image, [np.array(points, np.int32)], isClosed=True, color=color, thickness=2)

# ===== Hàm cắt vùng polygon =====
def crop_polygon_region(image, landmark_list, results, gray_background=True):
    h, w = image.shape[:2]
    points = []
    for idx in landmark_list:
        lm = results.multi_face_landmarks[0].landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

    region = cv2.bitwise_and(image, image, mask=mask)

    if gray_background:
        gray_bg = np.full_like(image, 128)
        region = np.where(mask[:, :, None] == 255, region, gray_bg)

    x, y, w_, h_ = cv2.boundingRect(np.array(points))
    pad = 5
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w_ + pad, w)
    y2 = min(y + h_ + pad, h)
    cropped = region[y1:y2, x1:x2]

    return cropped

# ===== Hàm tính diện tích polygon =====
def get_polygon_area(landmarks, ids, img_shape):
    h, w = img_shape[:2]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in ids]
    return cv2.contourArea(np.array(points, dtype=np.int32))

# ===== Hàm chính: cắt má trái, phải và cằm =====
def extract_face_regions(image_path, out_left=None, out_right=None, out_chin=None, debug_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print("Không đọc được ảnh:", image_path)
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("⚠️ Không phát hiện khuôn mặt:", image_path)
        return

    landmarks = results.multi_face_landmarks[0].landmark
    debug_img = image.copy()

    has_left = landmarks_valid(landmarks, LEFT_CHEEK_IDS, image.shape)
    has_right = landmarks_valid(landmarks, RIGHT_CHEEK_IDS, image.shape)

    cheek_left = cheek_right = None

    if has_left and has_right:
        area_left = get_polygon_area(landmarks, LEFT_CHEEK_IDS, image.shape)
        area_right = get_polygon_area(landmarks, RIGHT_CHEEK_IDS, image.shape)
        ratio = min(area_left, area_right) / max(area_left, area_right)

        if ratio < 0.5:
            if area_left > area_right:
                draw_polygon(debug_img, LEFT_CHEEK_IDS, results, (0, 255, 0))
                cheek_left = crop_polygon_region(image, LEFT_CHEEK_IDS, results)
                print("   Mặt nghiêng phải mạnh ➜ chỉ crop má trái")
            else:
                draw_polygon(debug_img, RIGHT_CHEEK_IDS, results, (0, 0, 255))
                cheek_right = crop_polygon_region(image, RIGHT_CHEEK_IDS, results)
                print("   Mặt nghiêng trái mạnh ➜ chỉ crop má phải")
        else:
            draw_polygon(debug_img, LEFT_CHEEK_IDS, results, (0, 255, 0))
            draw_polygon(debug_img, RIGHT_CHEEK_IDS, results, (0, 0, 255))
            cheek_left = crop_polygon_region(image, LEFT_CHEEK_IDS, results)
            cheek_right = crop_polygon_region(image, RIGHT_CHEEK_IDS, results)
            print("Mặt thẳng ➜ crop cả hai má")

    elif has_left:
        draw_polygon(debug_img, LEFT_CHEEK_IDS, results, (0, 255, 0))
        cheek_left = crop_polygon_region(image, LEFT_CHEEK_IDS, results)
        print("Chỉ crop má trái")

    elif has_right:
        draw_polygon(debug_img, RIGHT_CHEEK_IDS, results, (0, 0, 255))
        cheek_right = crop_polygon_region(image, RIGHT_CHEEK_IDS, results)
        print("  Chỉ crop má phải")

    if cheek_left is not None and out_left:
        cv2.imwrite(out_left, cheek_left)
    if cheek_right is not None and out_right:
        cv2.imwrite(out_right, cheek_right)

    # Cắt vùng cằm
    draw_polygon(debug_img, chin_indices, results, (255, 255, 0))
    chin_crop = crop_polygon_region(image, chin_indices, results)
    if out_chin:
        cv2.imwrite(out_chin, chin_crop)

    # Lưu ảnh debug nếu cần
    if debug_path:
        cv2.imwrite(debug_path, debug_img)

# ===== Chạy hàng loạt =====
def process_folder_all(input_folder, out_left_dir, out_right_dir, out_chin_dir, debug_dir):
    os.makedirs(out_left_dir, exist_ok=True)
    os.makedirs(out_right_dir, exist_ok=True)
    os.makedirs(out_chin_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for filename in tqdm(image_files, desc="Đang xử lý ảnh"):
        input_path = os.path.join(input_folder, filename)
        out_left = os.path.join(out_left_dir, filename)
        out_right = os.path.join(out_right_dir, filename)
        out_chin = os.path.join(out_chin_dir, filename)
        out_debug = os.path.join(debug_dir, filename)

        extract_face_regions(input_path, out_left, out_right, out_chin, out_debug)

# ===== GỌI THỬ =====
# input_image = "D:/KLTN/SKINTONE/pre_processing/data1/output_face_crop/light/968176.jpg"
# extract_face_regions(
#     input_image,
#     out_left="cheek_left.jpg",
#     out_right="cheek_right.jpg",
#     out_chin="chin_crop.jpg",
#     debug_path="debug_overlay.jpg"
# )

# ===== GỌI TRÊN THƯ MỤC =====
input_folder = "../pre_processing/data4/output_face_crop/mid-dark"
process_folder_all(
    input_folder,
    out_left_dir="../pre_processing/data4/output_crop_cheek_new/mid-dark/left",
    out_right_dir="../pre_processing/data4/output_crop_cheek_new/mid-dark/right",
    out_chin_dir="../pre_processing/data4/output_crop_chin_new/mid-dark",
    debug_dir="../pre_processing/data4/debug_overlay_new/mid-dark"
)
