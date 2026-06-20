from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"E:\ultralytics-main\ultralytics-main\runs\detect\train\weights\best.pt")
    model.val(split='test')