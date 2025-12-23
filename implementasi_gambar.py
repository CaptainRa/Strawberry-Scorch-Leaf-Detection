from ultralytics import YOLO
model = YOLO("best.pt")
model.predict(source="gambar.jpg", show=True, save=True, project="runs/detect", name="picture", conf=0.5)