import sys
import os
sys.path.append(r"E:\ultralytics-main\self_modules")
sys.path.append(r"E:\ultralytics-main\ultralytics-main\ultralytics")

from self_modules.Focal_EIOU import apply_focal_eiou_patch
apply_focal_eiou_patch()

from ultralytics import YOLO

def main():

    model = YOLO('YOLO8s_CA.yaml')


    model.load(r"E:\ultralytics-main\ultralytics-main\yolov8s.pt")

    results = model.train(
        data="data.yaml",
        epochs=200,
        imgsz=640,
        batch=6,
        device=0,
        workers=2,
        patience=50,
        mosaic=0.2,
        mixup=0.0,  # 禁用半透明重叠
        copy_paste=0.0,

        degrees=5.0,  # 极小幅度旋转兜底
        translate=0.05,
        scale=0.05,

        fliplr=0.5,
        flipud=0.0,  # 关闭上下翻转

        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,

        project=r'E:\ultralytics-main\runs\detect\herb_detection',
        name='v8s_final'
    )

if __name__ == '__main__':
    main()