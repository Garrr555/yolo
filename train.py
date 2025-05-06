from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Bisa ganti jadi yolov8s.pt untuk akurasi lebih tinggi
model.train(data="data.yaml", epochs=50, imgsz=640)
