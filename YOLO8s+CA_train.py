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

    # 3. 启动训练
    results = model.train(
        data=r"E:\ultralytics-main\data.yaml",
        epochs=200,  # 训练轮数
        imgsz=640,  # 保持 640
        batch=-1,
        workers=2,
        device=0,
        # === 核心：关闭/减弱过度增强 ===
        hsv_h=0.01,  # 色调抖动降到极低 (默认是 0.015)
        hsv_s=0.1,  # 饱和度抖动减弱 (默认是 0.7，太夸张了)
        hsv_v=0.1,  # 亮度抖动减弱 (默认是 0.4，导致过曝的元凶)

        mosaic=0.5,
        mixup=0.0,  # 确保关闭图像混合
        degrees=5.0,  # 允许轻微旋转（正负5度）
        translate=0.1,  # 允许轻微平移
        scale=0.2,  # 缩放比例不要太大，防止药材变太小
        fliplr=0.5,  # 允许左右翻转（这个对中药是安全的）
        flipud=0.5,# 允许上下翻转（安全）

        optimizer='auto', 
        patience=50,
        project='herb_detection',  # 训练结果保存的主文件夹名
        name='v8s_ca_focal_eiou_final'  # 本次实验的名称
    )

if __name__ == '__main__':
    main()