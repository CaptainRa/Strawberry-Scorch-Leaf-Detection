from ultralytics import YOLO
model = YOLO("best.pt")
model.predict(source="video2.mp4", show=True, save=True, project="runs/detect", name="video", conf=0.5)