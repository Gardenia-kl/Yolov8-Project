import sys
import os
sys.path.append(r"E:\ultralytics-main\self_modules")
sys.path.append(r"E:\ultralytics-main\ultralytics-main\ultralytics")

from self_modules.Focal_EIOU import apply_focal_eiou_patch
apply_focal_eiou_patch()

from ultralytics import YOLO



def main():

    model = YOLO('YOLO8s-Ghost-CA.yaml')


    model.load(r"E:\ultralytics-main\ultralytics-main\yolov8s.pt")

    # 3. 启动训练
    results = model.train(
        data=r"E:\ultralytics-main\data.yaml",
        epochs=150,
        imgsz=416,
        batch=-1,
        workers=3,

        mixup=0.15,
        # --- 基础优化策略 ---
        optimizer='AdamW',  # 轻量化模型推荐的优化器
        lr0=0.001,  # 初始学习率

        project='herb_detection',  # 训练结果保存的主文件夹名
        name='ghost_ca_focal_eiou_baseline'  # 本次实验的名称
    )

if __name__ == '__main__':
    main()