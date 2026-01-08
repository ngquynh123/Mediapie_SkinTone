# 🎨 Skin Tone Classification using MediaPipe & MobileNetV2

Phân loại tông màu da (skin tone) từ ảnh khuôn mặt sử dụng MediaPipe Face Mesh và MobileNetV2.

## 📋 Tổng quan

Dự án này phân loại tông màu da thành **6 loại** (Type_1 đến Type_6) dựa trên:

- Trích xuất vùng da mặt (má trái, má phải, cằm) bằng **MediaPipe Face Mesh**
- Loại bỏ nền và vùng mắt/miệng để chỉ giữ lại da
- Phân tích màu sắc trong không gian **LAB color space**
- Huấn luyện model phân loại bằng **MobileNetV2**

## 🏗️ Cấu trúc dự án

```
SKINTONE/
├── pre_processing/          # Tiền xử lý dữ liệu
│   ├── extract_face_regions.py      # Trích xuất vùng má & cằm
│   ├── extract_face_gray.py         # Trích xuất khuôn mặt (nền xám)
│   ├── lab_cheek_chin_data.py       # Xử lý dữ liệu LAB
│   ├── skin_tone_labeler.py         # Gán nhãn tông màu da
│   ├── augment_Type1.py             # Data augmentation
│   └── LAB.py                       # Phân tích LAB color space
│
├── public/                  # Training & Inference
│   ├── mobilenetV2.py               # Training script chính
│   ├── train_test_val.py            # Chia dữ liệu train/val/test
│   └── loc.py                       # Lọc ảnh theo LAB distance
│
├── mobilenetv2_best_*.pth   # Trained models (8 variants)
└── .gitignore
```

## 🚀 Cài đặt

### Yêu cầu

- Python 3.8+
- CUDA (nếu dùng GPU)

### Các thư viện

```bash
pip install torch torchvision
pip install mediapipe opencv-python
pip install scikit-image albumentations
pip install numpy pandas matplotlib seaborn tqdm
```

## 📊 Quy trình

### 1️⃣ Tiền xử lý dữ liệu

#### Trích xuất vùng má & cằm

```bash
python pre_processing/extract_face_regions.py
```

- Sử dụng MediaPipe Face Mesh phát hiện 478 landmarks
- Trích xuất 3 vùng: **má trái**, **má phải**, **cằm**
- Loại bỏ nền bằng Selfie Segmentation

#### Trích xuất khuôn mặt với nền xám

```bash
python pre_processing/extract_face_gray.py
```

- Giữ lại vùng khuôn mặt
- Che vùng mắt và miệng
- Thay nền bằng xám (RGB 128,128,128)

#### Phân tích LAB & gán nhãn

```bash
python pre_processing/skin_tone_labeler.py
```

- Chuyển đổi sang LAB color space
- Tính ΔE (Delta E) so với template
- Gán nhãn Type_1 → Type_6

### 2️⃣ Huấn luyện Model

```bash
python public/mobilenetV2.py
```

**Kiến trúc Model:**

- Base: MobileNetV2 (pretrained ImageNet)
- Custom classifier:
  ```
  Dropout(0.5) → Linear(1280, 128) → ReLU → BatchNorm1d → Linear(128, 6)
  ```

**Hyperparameters:**

- Image size: 224×224
- Batch size: 32
- Epochs: 50
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Scheduler: ReduceLROnPlateau
- Loss: CrossEntropyLoss

**Data Augmentation:**

- HorizontalFlip
- ShiftScaleRotate
- RandomBrightnessContrast
- GaussNoise

### 3️⃣ Đánh giá & Inference

```python
# Load model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Linear(128, 6)
)
model.load_state_dict(torch.load("mobilenetv2_best_cheek_chin.pth"))
model.eval()

# Predict
# ... (xem code trong public/mobilenetV2.py)
```

## 📁 Trained Models

Dự án cung cấp **8 model variants**:

| Model                                 | Mô tả                           |
| ------------------------------------- | ------------------------------- |
| `mobilenetv2_best.pth`                | Base model                      |
| `mobilenetv2_best_cheek_chin.pth`     | Train trên vùng má + cằm        |
| `mobilenetv2_best_skin.pth`           | Train trên toàn bộ da mặt       |
| `mobilenetv2_best_final.pth`          | Final optimized version         |
| `mobilenetv2_best_albu.pth`           | Với Albumentations augmentation |
| `mobilenetv2_best_f.pth`              | Fine-tuned variant              |
| `mobilenetv2_best_cheek.pth`          | Chỉ vùng má                     |
| `mobilenetv2_best_cheek_chin_new.pth` | Version mới nhất                |

## 🎯 Phương pháp phân loại

### A. Phân tích LAB Color Space

- **L**: Lightness (độ sáng) → phân biệt da sáng/tối
- **a**: Green-Red axis → màu đỏ trong da
- **b**: Blue-Yellow axis → màu vàng trong da

### B. Delta E (ΔE)

Đo khoảng cách màu sắc giữa 2 mẫu:

```
ΔE = √[(L1-L2)² + (a1-a2)² + (b1-b2)²]
```

- ΔE < 12: Gần với tone template
- ΔE > 20: Khác biệt rõ rệt

### C. Voting Mechanism

Dự đoán từ 3 vùng (má trái, má phải, cằm) → chọn kết quả phổ biến nhất

## 📈 Kết quả

- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- Test accuracy: ~75-80%

## 🔧 Tùy chỉnh

### Thay đổi số lớp phân loại

```python
num_classes = 4  # Thay 6 thành 4
model.classifier[-1] = nn.Linear(128, num_classes)
```

### Điều chỉnh threshold LAB

```python
# Trong loc.py
THRESHOLD = 10  # Giảm để chặt chẽ hơn
```

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repo
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Dự án này được phát hành dưới MIT License.

## 📧 Liên hệ

- GitHub: [@ngquynh123](https://github.com/ngquynh123)
- Repository: [Mediapie_SkinTone](https://github.com/ngquynh123/Mediapie_SkinTone)

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Face Mesh & Selfie Segmentation
- [PyTorch](https://pytorch.org/) - Deep Learning framework
- [MobileNetV2](https://arxiv.org/abs/1801.04381) - Efficient CNN architecture
