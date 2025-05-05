import os
from pathlib import Path
from yolov5 import detect

# Path ke folder uji
image_dir = Path('data/test_images')

# Ambil semua file gambar dari subfolder cats/ dan dogs/
image_paths = list(image_dir.rglob('*.jpg')) + list(image_dir.rglob('*.png'))

# Konversi ke list string path (YOLO butuh list path string atau glob)
image_paths = [str(p) for p in image_paths]

# Jalankan deteksi (contoh menggunakan torch.hub)
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
results = model(image_paths)

# Simpan hasil
results.save()  # akan tersimpan di runs/detect/
