from ultralytics import YOLO
import cv2
import os
from glob import glob

# Load model
model = YOLO("runs/detect/train2/weights/best.pt")

# Folder input dan output
input_folder = "test"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Ambil semua gambar JPG
image_paths = glob(os.path.join(input_folder, "*.jpg"))

# Cek jika tidak ada gambar
if not image_paths:
    raise FileNotFoundError("Tidak ada gambar .jpg di folder test/")

# Mapping ID ke label
names = model.names

# Loop setiap gambar
for img_path in image_paths:
    print(f"\nMendeteksi objek pada: {img_path}")
    results = model(img_path)

    # Hitung jumlah objek
    counts = {}
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            label = names[cls_id]
            counts[label] = counts.get(label, 0) + 1

    print("Jumlah objek terdeteksi:")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    # Simpan gambar hasil deteksi ke folder output/
    result_img = results[0].plot()
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, result_img)
    print(f"Hasil disimpan ke: {output_path}")
