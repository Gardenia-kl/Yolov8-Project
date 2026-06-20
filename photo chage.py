import os
import random
import shutil
import cv2
import albumentations as A

RAW_IMG_DIR = r"E:\ultralytics-main\ultralytics-main\images" # 你的原始图片存放路径
RAW_LABEL_DIR = r"E:\ultralytics-main\ultralytics-main\labels" # 你的原始标注txt存放路径
OUTPUT_DIR = "yolo_dataset"  # 最终生成的标准数据集根目录

# 划分比例 (训练集 : 验证集 : 测试集) 两者之和需为 1.0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 训练集的数据增强倍数（每张原始训练集图片额外生成几张增强图）
AUG_MULTIPLIER = 3



def get_train_pipeline():
    """配置最新的 Albumentations 增强流水线"""
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomResizedCrop(size=(640, 640), scale=(0.6, 1.0), ratio=(0.8, 1.2), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.MotionBlur(blur_limit=(3, 5), p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))


def read_yolo_labels(label_path):
    bboxes, class_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append([float(x) for x in parts[1:]])
                    class_labels.append(int(parts[0]))
    return bboxes, class_labels


def write_yolo_labels(output_path, bboxes, class_labels):
    with open(output_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def augment_and_save_train(img_path, lbl_path, out_img_dir, out_label_dir, base_name):
    """专门针对训练集：保存原图并生成增强图"""
    image = cv2.imread(img_path)
    if image is None: return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = read_yolo_labels(lbl_path)

    # 1. 保存原始训练图和标签
    cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_orig.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    write_yolo_labels(os.path.join(out_label_dir, f"{base_name}_orig.txt"), bboxes, class_labels)

    # 2. 循环生成多张增强图
    transform = get_train_pipeline()
    for i in range(AUG_MULTIPLIER):
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            if len(transformed['bboxes']) == 0 and len(bboxes) > 0:
                continue  # 避免生成无目标的空样本

            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_aug_{i}.jpg"),
                        cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            write_yolo_labels(os.path.join(out_label_dir, f"{base_name}_aug_{i}.txt"), transformed['bboxes'],
                              transformed['class_labels'])
        except Exception as e:
            print(f"警告：图片 {base_name} 在执行第 {i} 次增强时发生错误: {e}")


def main():
    # 1. 目录初始化
    sub_dirs = ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']
    for sd in sub_dirs:
        os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

    # 2. 获取并对齐所有数据
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(RAW_IMG_DIR) if f.lower().endswith(img_extensions)]

    valid_pairs = []
    for img_name in all_images:
        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + '.txt'
        lbl_path = os.path.join(RAW_LABEL_DIR, lbl_name)
        if os.path.exists(lbl_path):
            valid_pairs.append((img_name, lbl_name, base_name))

    print(f"成功匹配到有效图片与标签对: {len(valid_pairs)} 组")

    # 3. 随机打乱数据集
    random.seed(42)  # 固定随机种子以保证结果可复现
    random.shuffle(valid_pairs)

    # 4. 计算切分索引
    total = len(valid_pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    # 5. 开始分发与处理
    print("正在进行数据集划分与处理...")
    for idx, (img_name, lbl_name, base_name) in enumerate(valid_pairs):
        src_img = os.path.join(RAW_IMG_DIR, img_name)
        src_lbl = os.path.join(RAW_LABEL_DIR, lbl_name)

        if idx < train_end:
            # 训练集：需要复制并执行离线数据增强
            out_img_dir = os.path.join(OUTPUT_DIR, 'images/train')
            out_lbl_dir = os.path.join(OUTPUT_DIR, 'labels/train')
            augment_and_save_train(src_img, src_lbl, out_img_dir, out_lbl_dir, base_name)

        elif idx < val_end:
            # 验证集：只复制原始文件，绝不增强
            shutil.copy(src_img, os.path.join(OUTPUT_DIR, 'images/val', f"{base_name}.jpg"))
            shutil.copy(src_lbl, os.path.join(OUTPUT_DIR, 'labels/val', f"{base_name}.txt"))

        else:
            # 测试集：只复制原始文件，绝不增强
            shutil.copy(src_img, os.path.join(OUTPUT_DIR, 'images/test', f"{base_name}.jpg"))
            shutil.copy(src_lbl, os.path.join(OUTPUT_DIR, 'labels/test', f"{base_name}.txt"))

    print(f"\n数据集划分与增强全部完成！最终保存在路径: {OUTPUT_DIR}")
    print(f"训练集原始: {train_end}张 -> 增强扩充后总计: {len(os.listdir(os.path.join(OUTPUT_DIR, 'images/train')))}张")
    print(f"验证集原始: {int(total * VAL_RATIO)}张")
    print(f"测试集原始: {total - val_end}张")


if __name__ == "__main__":
    main()