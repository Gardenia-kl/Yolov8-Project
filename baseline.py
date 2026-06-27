from ultralytics import YOLO

if __name__ == "__main__":
    model= YOLO(r"E:\ultralytics-main\ultralytics-main\yolov8s.pt")
    model.train(
        data=r"E:\ultralytics-main\data.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        cache=False,
        workers=2,
    )