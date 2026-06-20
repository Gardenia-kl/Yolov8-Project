import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

def main():

    model = YOLO('YOLO8s-Ghost-CA.yaml')


    model.load(r"E:\ultralytics-main\ultralytics-main\yolov8s.pt")

    # 3. 启动训练
    results = model.train(
        data=r"E:\ultralytics-main\data.yaml",
        epochs=150,
        imgsz=416,
        batch=-1,
        workers=2,


        # --- 基础优化策略 ---
        optimizer='AdamW',  # 轻量化模型推荐的优化器
        lr0=0.001,  # 初始学习率

        project='herb_detection',  # 训练结果保存的主文件夹名
        name='ghost_ca_baseline'  # 本次实验的名称
    )

if __name__ == '__main__':
    main()