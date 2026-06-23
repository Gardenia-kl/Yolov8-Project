import os
import random
import shutil
import cv2
import albumentations as A

# ================= 1. 双输入源路径配置 =================
# 老数据（群落大堆图）：约900+张
OLD_IMG_DIR = r"E:\ultralytics-main\images_old"
OLD_LABEL_DIR = r"E:\ultralytics-main\labels_old"

# 新数据（单片孤立图）：31张
NEW_IMG_DIR = r"E:\ultralytics-main\images_new"
NEW_LABEL_DIR = r"E:\ultralytics-main\labels_new"

OUTPUT_DIR = "yolo_dataset"  # 最终生成的标准数据集根目录

# 数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 【抗稀释核心倍率】
OLD_MULTIPLIER = 2  # 老数据：放大 2 倍 (生成2张增强图)
NEW_MULTIPLIER = 30  # 新数据：疯狂放大 30 倍！彻底解决过拟合单片漏检问题


# ======================================================

def get_train_pipeline():
    """专为白纸环境定制的空间与噪声增强流水线"""
    return A.Compose([
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),

        # 旋转平移，缺失部分用白色(255,255,255)填充，完美融入白纸背景
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=360,
                           border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.4),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


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


def augment_and_save_train(img_path, lbl_path, out_img_dir, out_label_dir, base_name, multiplier):
    """根据传入的倍数进行增强并保存"""
    image = cv2.imread(img_path)
    if image is None: return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_labels = read_yolo_labels(lbl_path)

    # 保存原图
    cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_orig.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    write_yolo_labels(os.path.join(out_label_dir, f"{base_name}_orig.txt"), bboxes, class_labels)

    # 循环增强
    transform = get_train_pipeline()
    for i in range(multiplier):
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            if len(transformed['bboxes']) == 0 and len(bboxes) > 0:
                continue

            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_aug_{i}.jpg"),
                        cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
            write_yolo_labels(os.path.join(out_label_dir, f"{base_name}_aug_{i}.txt"), transformed['bboxes'],
                              transformed['class_labels'])
        except Exception as e:
            pass  # 忽略偶发的越界错误


def load_dataset_pairs(img_dir, lbl_dir, data_type):
    """从指定文件夹加载所有图文对，并打上数据类型标签"""
    pairs = []
    if not os.path.exists(img_dir): return pairs

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]

    for img_name in all_images:
        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_name)
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(lbl_path):
            # 将路径和数据类型（old/new）一起存入字典
            pairs.append({
                'img_path': img_path,
                'lbl_path': lbl_path,
                'base_name': base_name,
                'type': data_type
            })
    return pairs


def main():
    # 初始化输出目录
    sub_dirs = ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']
    for sd in sub_dirs:
        os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

    # 分别加载老数据和新数据
    old_pairs = load_dataset_pairs(OLD_IMG_DIR, OLD_LABEL_DIR, 'old')
    new_pairs = load_dataset_pairs(NEW_IMG_DIR, NEW_LABEL_DIR, 'new')

    print(f"✅ 读取成功: 老数据 {len(old_pairs)} 组，新数据 {len(new_pairs)} 组")

    # 把新老数据合并成一个大池子
    all_pairs = old_pairs + new_pairs

    # 彻底打乱（公平混洗）
    random.seed(42)
    random.shuffle(all_pairs)

    total = len(all_pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    print("🚀 正在执行智能混合与定向倍率增强...")

    old_train_cnt, new_train_cnt = 0, 0

    for idx, item in enumerate(all_pairs):
        src_img = item['img_path']
        src_lbl = item['lbl_path']
        base_name = item['base_name']

        if idx < train_end:
            # 根据字典里的类型标签，赋予绝对精确的增强倍数
            if item['type'] == 'new':
                multiplier = NEW_MULTIPLIER
                new_train_cnt += 1
            else:
                multiplier = OLD_MULTIPLIER
                old_train_cnt += 1

            out_img_dir = os.path.join(OUTPUT_DIR, 'images/train')
            out_lbl_dir = os.path.join(OUTPUT_DIR, 'labels/train')
            augment_and_save_train(src_img, src_lbl, out_img_dir, out_lbl_dir, base_name, multiplier)

        elif idx < val_end:
            shutil.copy(src_img, os.path.join(OUTPUT_DIR, 'images/val', f"{base_name}.jpg"))
            shutil.copy(src_lbl, os.path.join(OUTPUT_DIR, 'labels/val', f"{base_name}.txt"))
        else:
            shutil.copy(src_img, os.path.join(OUTPUT_DIR, 'images/test', f"{base_name}.jpg"))
            shutil.copy(src_lbl, os.path.join(OUTPUT_DIR, 'labels/test', f"{base_name}.txt"))

    print("\n🎉 数据集处理完成！")
    print(
        f"实际进入训练集：老数据 {old_train_cnt} 张 (放大 {OLD_MULTIPLIER} 倍)，新数据 {new_train_cnt} 张 (放大 {NEW_MULTIPLIER} 倍)")
    print(f"最终生成的 YOLO 训练集包含海量数据，请前往 {OUTPUT_DIR} 目录查看！")


if __name__ == "__main__":
    main()