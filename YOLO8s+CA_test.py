from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"E:\ultralytics-main\runs\detect\herb_detection\v8s_final\weights\best.pt")
    model.val(
        data=r"E:\ultralytics-main\data.yaml",
        split='test',   )