import pandas as pd

# ===== Hàm gộp các vùng (má trái, phải, cằm) lại theo tên ảnh =====
def group_mean_lab_by_image(df):
    """
    Gộp các vùng da (cheek trái/phải và chin) lại theo image → lấy trung bình L, a, b
    """
    grouped = df.groupby("image").agg({
        "L": "mean",
        "a": "mean",
        "b": "mean"
    }).reset_index()

    # Đặt lại tên cột
    grouped.columns = ["image", "L_mean", "a_mean", "b_mean"]
    return grouped

# ===== MAIN - Gọi và lưu file trung bình =====
if __name__ == "__main__":
    input_csv = "lab_cheek_chin_data.csv"        # File đã tạo trước
    output_csv = "lab_face_avg.csv"              # File sẽ xuất ra

    try:
        df = pd.read_csv(input_csv)
        df_avg = group_mean_lab_by_image(df)

        df_avg.to_csv(output_csv, index=False)
        print(f"✅ Đã lưu file trung bình toàn mặt: {output_csv}")
        print(df_avg.head())

    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {input_csv}")
