"""
数据集分类：H5 读取 → 训练/验证拆分 → YOLO 格式导出
=====================================================
流程:
  1. GPU / 环境检查
  2. 读取 H5 中的 train&val 图像和标签
  3. 分层随机拆分训练集 / 验证集 (8:2)
  4. 导出 train&val 图像 + YOLO 标签
  5. 导入测试集 (保留原始文件名)
  6. 创建 dataset.yaml
"""

import shutil
from pathlib import Path
import numpy as np
import h5py
import yaml
from PIL import Image

# ======================== 路径配置 ========================
BASE_DIR = Path(__file__).parent  # 当前脚本所在目录
DATASETS_DIR = BASE_DIR /"Data_Prep"/"RFDD_clean"
WORK_DIR = BASE_DIR /"Data_Prep"/"RFDD_Grouping"
WORK_DIR.mkdir(exist_ok=True)

# 输入文件
TRAINVAL_IMG_H5 = DATASETS_DIR / "train&val" / "images" / "RFDD_Train&val_Images_dedup.h5"
TRAINVAL_LBL_H5 = DATASETS_DIR / "train&val" / "labels" / "RFDD_Train&val_Labels_dedup.h5"

# YOLO 格式输出目录
YOLO_ROOT = WORK_DIR
YOLO_IMAGES_TRAIN = YOLO_ROOT / "images" / "train"
YOLO_IMAGES_VAL = YOLO_ROOT / "images" / "val"
YOLO_IMAGES_TEST = YOLO_ROOT / "images" / "test"
YOLO_LABELS_TRAIN = YOLO_ROOT / "labels" / "train"
YOLO_LABELS_VAL = YOLO_ROOT / "labels" / "val"
YOLO_LABELS_TEST = YOLO_ROOT / "labels" / "test"

CLASS_NAMES_EN = ["Deformed", "Fractured", "Missing", 
                  "Inverted", "Normal", "Displaced"]
NUM_CLASSES = 6

# 训练/验证拆分比例
TRAIN_RATIO = 0.8 # 80% 训练，20% 验证
RANDOM_SEED = 42  #固定随机种子，确保每次运行结果一致


# ===================== 1. 目录结构 =====================

def create_dir_structure():
    """创建 YOLO 格式目录结构，清空已有文件"""
    for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL, YOLO_IMAGES_TEST,
              YOLO_LABELS_TRAIN, YOLO_LABELS_VAL, YOLO_LABELS_TEST]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()


# ===================== 3. 数据导出 (H5 → YOLO 格式) =====================

def export_h5_to_yolo():
    """
    将 H5 中的 train&val 图像和标签导出为 YOLO 格式:
      images/train/*.png  +  labels/train/*.txt
      images/val/*.png    +  labels/val/*.txt
    测试集从 PNG + TXT 复制到统一目录结构，保留原始文件名
    """
    print("\n" + "=" * 60)
    print("  数据导出: H5 → YOLO 格式")
    print("=" * 60)

    create_dir_structure() # 创建目录结构并清空旧文件

    # ----- 读取 H5 -----
    print("  读取 H5 文件...")
    with h5py.File(TRAINVAL_IMG_H5, 'r') as f:
        # 该处直接读取整个数据集到内存一共 1250 张，占用5GB内存
        images = f['images'][:]   # (1250, 2021, 2048) uint8 
        filenames = [fn.decode('utf-8', errors='replace') if isinstance(fn, bytes) else str(fn)
                     for fn in f['filenames'][:]]

    with h5py.File(TRAINVAL_LBL_H5, 'r') as f:
        labels = f['labels'][:]   # (1250, 6, 5) float32

    print(f"  图像: {images.shape}, 标签: {labels.shape}")

    # ----- 分层随机拆分 train/val (8:2，纯 numpy 实现) -----
    # 策略: 分别对"含缺陷"和"纯正常"两组进行随机拆分，保持比例一致
    sample_class_cnt = np.zeros(len(labels))
    for i in range(len(labels)):
        valid = labels[i][labels[i][:, 0] >= 0]
        sample_class_cnt[i] = (valid[:, 0].astype(int) != 4).sum()
    has_defect = sample_class_cnt > 0

    def stratified_split(indices, stratify_array, test_ratio, seed):
        """手动分层拆分: 每层内按比例随机分配 train/val"""
        rng = np.random.RandomState(seed)
        train_idx, val_idx = [], []
        for strat_val in np.unique(stratify_array):
            stratum_indices = indices[stratify_array == strat_val]
            rng.shuffle(stratum_indices)
            n_val = max(1, int(len(stratum_indices) * test_ratio))
            val_idx.append(stratum_indices[:n_val])
            train_idx.append(stratum_indices[n_val:])
        return np.concatenate(train_idx), np.concatenate(val_idx)

    train_idx, val_idx = stratified_split(
        np.arange(len(labels)), has_defect, 1 - TRAIN_RATIO, RANDOM_SEED
    )
    train_set = set(train_idx)
    val_set = set(val_idx)
    print(f"  训练: {len(train_idx)} 张,  验证: {len(val_idx)} 张")

    # ----- 导出 train&val 图像和标签 -----
    print("  导出 Train&Val 图像和标签...")
    for i in range(len(images)):
        if i in train_set:
            img_dir, lbl_dir = YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN
        else:
            img_dir, lbl_dir = YOLO_IMAGES_VAL, YOLO_LABELS_VAL

        img_name = filenames[i]
        if not img_name.endswith('.png'):
            img_name += '.png'

        # 灰度 → RGB 三通道，保存为 PNG
        gray_img = images[i]
        rgb_img = np.stack([gray_img] * 3, axis=-1)
        Image.fromarray(rgb_img).save(img_dir / img_name)

        # YOLO 标签: class_id cx cy w h
        valid_boxes = labels[i][labels[i][:, 0] >= 0]
        txt_path = lbl_dir / img_name.replace('.png', '.txt')
        with open(txt_path, 'w') as f:
            for box in valid_boxes:
                cls_id, cx, cy, w, h = box
                f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"  Train: {len(train_idx)} 张 → {YOLO_IMAGES_TRAIN}")
    print(f"  Val:   {len(val_idx)} 张 → {YOLO_IMAGES_VAL}")

    # ----- 测试集 (PNG + TXT) → 统一目录（保留原始文件名） -----
    print("  导出 Test 数据...")
    test_img_src = BASE_DIR / "RFDD_datasets" / "test" / "images"
    test_lbl_src = BASE_DIR / "RFDD_datasets" / "test" / "labels"

    test_count = 0
    for png_file in sorted(test_img_src.glob("*.png")):
        test_count += 1

        img = Image.open(png_file).convert("RGB")
        img.save(YOLO_IMAGES_TEST / png_file.name)

        lbl_src = test_lbl_src / (png_file.stem + ".txt")
        lbl_dst = YOLO_LABELS_TEST / (png_file.stem + ".txt")
        if lbl_src.exists():
            shutil.copy(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("")

    print(f"  Test: {test_count} 张 → {YOLO_IMAGES_TEST}")
    return len(train_idx), len(val_idx), test_count


# ===================== 4. 创建 YAML =====================

def create_yaml():
    """创建 dataset.yaml"""
    yaml_path = YOLO_ROOT / "RFDD.yaml"
    data_yaml = {
        'path': str(YOLO_ROOT.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': NUM_CLASSES,
        'names': {i: name for i, name in enumerate(CLASS_NAMES_EN)},
    }
    yaml_path.write_text(yaml.dump(data_yaml, default_flow_style=False, allow_unicode=True),
                         encoding='utf-8')
    print(f"\n  YAML 配置: {yaml_path}")
    return yaml_path


# ===================== 主流程 =====================

def main():
    print("=" * 60)
    print("  数据集分类 — H5 读取 & 数据导出")
    print("=" * 60)

    # 1. H5 → YOLO 导出
    n_train, n_val, n_test = export_h5_to_yolo()

    # 2. 创建 YAML
    yaml_path = create_yaml()

    # 汇总
    print(f"\n  {'='*60}")
    print(f"  数据集统计")
    print(f"  {'='*60}")
    print(f"  训练集:       {n_train} 张")
    print(f"  验证集:       {n_val} 张")
    print(f"  测试集:       {n_test} 张")
    print(f"  类别数:       {NUM_CLASSES}")
    print(f"  类别:         {', '.join(CLASS_NAMES_EN)}")
    print(f"  拆分比例:     {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}")
    print(f"  随机种子:     {RANDOM_SEED}")
    print(f"  YAML:         {yaml_path}")
    print(f"\n  数据集分类完成! 可运行 yolo模型构建.py 开始训练。")


if __name__ == "__main__":
    main()
