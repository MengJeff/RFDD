"""
任务 1：数据准备与可视化
-----------------------
流程:
  1. 读取训练集、验证集和测试集
  2. 可视化至少 10 张图像的标注框，检查类别与框位置
  3. 统计每一类目标数量，绘制类别分布图
  4. 分析类别不平衡、小目标、相似类别等问题
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
# ---------- 中文字体设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 路径 ----------
BASE_DIR = Path(__file__).parent 
# 训练/验证集 H5 文件路径
TRAINVAL_IMG_H5 = BASE_DIR / "Data_Prep" / "RFDD_clean"/ "train&val" / "images" / "RFDD_Train&val_Images_dedup.h5"
# 训练/验证集标签 H5 文件路径
TRAINVAL_LBL_H5 = BASE_DIR / "Data_Prep" / "RFDD_clean"/ "train&val" / "labels" / "RFDD_Train&val_Labels_dedup.h5"
# 测试集图像和标签路径
TEST_IMG_DIR = BASE_DIR / "RFDD_datasets" / "test" / "images"
TEST_LBL_DIR = BASE_DIR / "RFDD_datasets"/ "test" / "labels"
# 输出目录
OUTPUT_DIR = Path(__file__).parent / "outputs_task1"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    0: "变形 (Deformed)",
    1: "断裂 (Fractured)",
    2: "缺失 (Missing)",
    3: "反装 (Inverted)",
    4: "正常 (Normal)",
    5: "位移 (Displaced)",
}
NUM_CLASSES = 6

# ========================== 1. 数据加载 ==========================

def load_trainval_h5():
    """从 H5 文件读取 train&val 图像与标签"""
    print("=" * 55)
    print("  正在读取 Train&Val 数据（去重后 H5）...")

    with h5py.File(TRAINVAL_IMG_H5, 'r') as f:
        images = f['images'][:]          # (1250, 2021, 2048) uint8
        filenames = f['filenames'][:]    # (1250,) bytes

    with h5py.File(TRAINVAL_LBL_H5, 'r') as f:
        labels = f['labels'][:]          # (1250, 6, 5) float32
        lbl_filenames = f['filenames'][:]

    print(f"  图像: {images.shape} (N, H, W), dtype={images.dtype}")
    print(f"  标签: {labels.shape} (N, 6, 5), dtype={labels.dtype}")
    return images, labels, filenames


def load_test_png():
    """从 PNG + TXT 读取测试集数据（100 张）"""
    print("\n  正在读取 Test 数据（PNG + TXT）...")

    test_images, test_labels, test_fns = [], [], []

    for img_path in sorted(TEST_IMG_DIR.glob("*.png")):
        base = img_path.stem
        lbl_path = TEST_LBL_DIR / f"{base}.txt"

        # 读取图像（plt.imread 对 PNG 返回 float [0,1]，需乘 255 转为 uint8）
        img = plt.imread(str(img_path))           # (H, W) or (H, W, 3)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)    # float [0,1] → uint8 [0,255]
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)  # RGB → 灰度
        test_images.append(img)

        # 读取标签（YOLO 格式：每行 class_id cx cy w h）
        boxes = []
        if lbl_path.exists():
            with open(lbl_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = map(float, parts)
                        boxes.append([cls, cx, cy, w, h])

        # 补齐到 6 个框（与 train&val 格式一致，不足用 -1 填充）
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5))
        if len(boxes) < 6:
            pad = np.full((6 - len(boxes), 5), -1.0, dtype=np.float32)
            boxes = np.vstack([boxes, pad]) if len(boxes) > 0 else pad
        test_labels.append(boxes)
        test_fns.append(base)

    t_images = np.array(test_images, dtype=object)  # 尺寸不一致用 object
    t_labels = np.array(test_labels, dtype=np.float32)
    print(f"  测试图像: {len(test_images)} 张")
    print(f"  测试标签: {t_labels.shape}")
    return t_images, t_labels, test_fns


def split_trainval(labels, filenames):
    """
    将 train&val 按 80:20 分层拆分为训练集和验证集。
    用于后续模型训练（本任务主要做统计，暂不生成新的 H5，仅做索引号记录）。
    """
    # 取每个样本中有效框的类别分布（排除 class_id == -1 的填充位）
    sample_class_counts = []
    for i in range(len(labels)):
        valid = labels[i][labels[i][:, 0] >= 0]
        counts = np.bincount(valid[:, 0].astype(int), minlength=NUM_CLASSES)
        sample_class_counts.append(counts)
    sample_class_counts = np.array(sample_class_counts)

    # 按是否存在某个缺陷类做简单分层
    has_defect = (sample_class_counts[:, :4].sum(axis=1) + sample_class_counts[:, 5]) > 0

    # 手动分层拆分 (替代 sklearn train_test_split)
    indices = np.arange(len(labels))
    rng = np.random.RandomState(42)
    train_idx_list, val_idx_list = [], []
    for strat_val in np.unique(has_defect):
        stratum_idx = indices[has_defect == strat_val]
        rng.shuffle(stratum_idx)
        n_val = max(1, int(len(stratum_idx) * 0.2))
        val_idx_list.append(stratum_idx[:n_val])
        train_idx_list.append(stratum_idx[n_val:])
    train_idx = np.concatenate(train_idx_list)
    val_idx = np.concatenate(val_idx_list)
    print(f"\n  Train&Val 拆分:")
    print(f"    训练集: {len(train_idx)} 张, 验证集: {len(val_idx)} 张")
    return train_idx, val_idx


# ========================== 2. 标注框可视化 ==========================

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """YOLO 归一化坐标 → 左上/右下像素坐标"""
    x1 = max(0, (cx - w / 2) * img_w)
    y1 = max(0, (cy - h / 2) * img_h)
    x2 = min(img_w, (cx + w / 2) * img_w)
    y2 = min(img_h, (cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes(ax, labels_per_img, img_h, img_w):
    """在 ax 上绘制所有标注框"""
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']
    for i in range(len(labels_per_img)):
        cls_id, cx, cy, w, h = labels_per_img[i]
        if cls_id < 0 or (w == 0 and h == 0):  # 跳过填充位
            continue
        cls_id = int(cls_id)
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=colors[cls_id % len(colors)], facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 6), CLASS_NAMES[cls_id],
                fontsize=7, color=colors[cls_id % len(colors)],
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))


def visualize_samples(images, labels, filenames, dataset_name, num_vis=10,
                      output_prefix="vis"):
    """
    随机选取 num_vis 张图像并可视化标注框。
    对 2021×2048 的大图先降采样到便于显示的大小。
    """
    indices = np.random.choice(len(labels), size=min(num_vis, len(labels)), replace=False)
    cols = 5
    rows = int(np.ceil(len(indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]
        # 从 object 数组中取出实际图像并确保是 numpy 数组
        img = np.asarray(images[idx], dtype=np.uint8)
        img_h, img_w = img.shape[:2]
        # 降采样显示
        scale = min(1.0, 800 / max(img_h, img_w))
        stride = max(1, int(1 / scale))
        display = img[::stride, ::stride]
        ax.imshow(display, cmap='gray', vmin=0, vmax=255)

        # 按比例调整框坐标
        lbl = np.array(labels[idx])
        orig_h, orig_w = img_h, img_w
        disp_h, disp_w = display.shape[:2]
        for j in range(len(lbl)):
            cls_id, cx, cy, w, h = lbl[j]
            if cls_id < 0 or (w == 0 and h == 0):
                continue
            cls_id = int(cls_id)
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, orig_w, orig_h)
            # 映射到缩小后的图像尺寸
            x1 *= disp_w / orig_w; y1 *= disp_h / orig_h
            x2 *= disp_w / orig_w; y2 *= disp_h / orig_h
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=plt.cm.tab10(cls_id), facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 4), CLASS_NAMES[cls_id][:2],
                    fontsize=6, color='red', weight='bold')
        ax.set_title(f"{dataset_name} #{idx}", fontsize=9)
        ax.axis('off')

    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{dataset_name} — 标注框可视化 ({len(indices)} 张)", fontsize=14)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{output_prefix}_{dataset_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存可视化: {out_path}")


# ========================== 3. 类别统计与分布图 ==========================

def count_classes(labels_list, dataset_name):
    """统计每个类别的目标数量（排除填充位 class_id < 0）"""
    counts = np.zeros(NUM_CLASSES, dtype=int)
    total_boxes = 0
    for lbl in labels_list:
        valid = lbl[lbl[:, 0] >= 0]
        if len(valid) == 0:
            continue
        cls_ids = valid[:, 0].astype(int)
        for c in cls_ids:
            if 0 <= c < NUM_CLASSES:
                counts[c] += 1
                total_boxes += 1
    return counts, total_boxes


def print_class_table(train_cnt, val_cnt, test_cnt, train_total, val_total, test_total):
    """打印类别统计表"""
    header = f"{'类别':<22s} {'训练集':>8s} {'验证集':>8s} {'测试集':>8s} {'合计':>8s} {'占比':>8s}"
    print("\n" + "=" * 75)
    print("  类别数量统计表")
    print("=" * 75)
    print(header)
    print("-" * 75)

    all_cnt = train_cnt + val_cnt + test_cnt
    grand_total = all_cnt.sum()
    for cid in range(NUM_CLASSES):
        name = CLASS_NAMES[cid]
        print(f"  {name:<22s} {train_cnt[cid]:>8d} {val_cnt[cid]:>8d} "
              f"{test_cnt[cid]:>8d} {all_cnt[cid]:>8d} {all_cnt[cid]/grand_total*100:>7.1f}%")

    print("-" * 75)
    print(f"  {'合计':<22s} {train_total:>8d} {val_total:>8d} "
          f"{test_total:>8d} {grand_total:>8d} {100.0:>7.1f}%")
    print("=" * 75)


def plot_class_distribution(train_cnt, val_cnt, test_cnt, save_name="class_distribution.png"):
    """绘制类别分布柱状图"""
    x = np.arange(NUM_CLASSES)
    width = 0.25
    names_short = ["变形\nDeformed", "断裂\nFractured", "缺失\nMissing",
                   "反装\nInverted", "正常\nNormal", "位移\nDisplaced"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, train_cnt, width, label='训练集 (Train)', color='#5470c6')
    bars2 = ax.bar(x, val_cnt, width, label='验证集 (Val)', color='#91cc75')
    bars3 = ax.bar(x + width, test_cnt, width, label='测试集 (Test)', color='#fac858')

    ax.set_xlabel('类别', fontsize=13)
    ax.set_ylabel('目标数量', fontsize=13)
    ax.set_title('类别分布柱状图 (Class Distribution)', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names_short, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 3,
                        str(int(h)), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / save_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  已保存类别分布图: {out_path}")

    # 额外画一张不含 Normal 的放大图（因为 Normal 数量太大）
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    defect_classes = [0, 1, 2, 3, 5]
    x2 = np.arange(len(defect_classes))
    names_defect = [names_short[i] for i in defect_classes]

    train_d = [train_cnt[i] for i in defect_classes]
    val_d = [val_cnt[i] for i in defect_classes]
    test_d = [test_cnt[i] for i in defect_classes]

    ax2.bar(x2 - width, train_d, width, label='训练集', color='#5470c6')
    ax2.bar(x2, val_d, width, label='验证集', color='#91cc75')
    ax2.bar(x2 + width, test_d, width, label='测试集', color='#fac858')

    ax2.set_title('缺陷类分布（不含正常类）', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names_defect, fontsize=11)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_path2 = OUTPUT_DIR / "class_distribution_defects_only.png"
    plt.savefig(out_path2, dpi=150)
    plt.close()
    print(f"  已保存缺陷类分布图: {out_path2}")


# ========================== 4. 数据集问题分析 ==========================

def analyze_dataset(train_labels, val_labels, test_labels, all_trainval_labels):
    """综合分析数据集特征与问题"""
    print("\n" + "=" * 65)
    print("  数据集问题分析")
    print("=" * 65)

    # --- A. 类别不平衡 ---
    train_cnt, _ = count_classes(train_labels, "Train")
    val_cnt, _ = count_classes(val_labels, "Val")
    test_cnt, _ = count_classes(test_labels, "Test")
    all_cnt = train_cnt + val_cnt + test_cnt

    print("\n  【1. 类别不平衡 (Class Imbalance)】")
    print(f"    正常类 (Normal) 占比: {all_cnt[4] / all_cnt.sum() * 100:.1f}%")
    print(f"    5 种缺陷类合计占比:    {(all_cnt.sum() - all_cnt[4]) / all_cnt.sum() * 100:.1f}%")
    print(f"    正常/缺陷比例约:        {all_cnt[4] / max(1, (all_cnt.sum() - all_cnt[4])):.1f} : 1")
    print("    结论: 存在极端类别不平衡，正常类占绝大多数。")
    print("    建议: 使用加权损失函数 (如 Focal Loss) 或重采样策略。")

    # --- B. 小目标问题 ---
    # 将 train&val 和 test 标签展平，排除填充位（class_id < 0）
    trainval_flat = all_trainval_labels.reshape(-1, 5)
    test_flat = test_labels.reshape(-1, 5)
    all_boxes = np.vstack([trainval_flat, test_flat])
    all_boxes = all_boxes[all_boxes[:, 0] >= 0]     # 只保留有效框
    areas = all_boxes[:, 3] * all_boxes[:, 4]        # w * h (归一化面积)

    small_thresh = 0.01   # 目标面积 < 1% 图像
    medium_thresh = 0.05  # 目标面积 < 5%
    pct_small = (areas < small_thresh).sum() / len(areas) * 100
    pct_medium = ((areas >= small_thresh) & (areas < medium_thresh)).sum() / len(areas) * 100
    pct_large = (areas >= medium_thresh).sum() / len(areas) * 100

    print(f"\n  【2. 目标尺寸分析 (Object Size)】")
    print(f"    标注框面积均值: {areas.mean():.4f}  (占图像比例)")
    print(f"    标注框面积中位数: {np.median(areas):.4f}")
    print(f"    面积 < 1% (极小):  {pct_small:.1f}%")
    print(f"    面积 1%-5% (小):    {pct_medium:.1f}%")
    print(f"    面积 >= 5% (中大):  {pct_large:.1f}%")

    if pct_small > 10 or pct_medium > 30:
        print("    结论: 存在较明显的小目标问题。紧固件在 2021×2048 的大图中占比小。")
        print("    建议: 适当缩放输入分辨率，或用多尺度训练策略。")
    else:
        print("    结论: 目标尺寸比例尚可，但需结合具体锚框设计。")

    # 画目标面积分布直方图
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.hist(areas * 100, bins=80, edgecolor='white', color='#6e8bc3', alpha=0.85)
    ax3.axvline(x=1, color='red', linestyle='--', label='1% (极小阈值)')
    ax3.axvline(x=5, color='orange', linestyle='--', label='5% (小目标阈值)')
    ax3.set_xlabel('标注框面积（占图像 %）', fontsize=12)
    ax3.set_ylabel('频数', fontsize=12)
    ax3.set_title('标注框面积分布直方图 (BBox Area Distribution)', fontsize=13)
    ax3.legend()
    plt.tight_layout()
    out_path3 = OUTPUT_DIR / "bbox_area_histogram.png"
    plt.savefig(out_path3, dpi=150)
    plt.close()
    print(f"  已保存目标尺寸分布图: {out_path3}")

    # --- C. 每个样本的目标数量 ---
    sample_nums = []
    for lbl in all_trainval_labels:
        valid = lbl[lbl[:, 0] >= 0]
        sample_nums.append(len(valid))
    sample_nums = np.array(sample_nums)
    print(f"\n  【3. 每图目标数量】")
    print(f"    最小值: {sample_nums.min()}, 最大值: {sample_nums.max()}")
    print(f"    均值: {sample_nums.mean():.1f}, 中位数: {np.median(sample_nums):.0f}")
    unique_n, counts_n = np.unique(sample_nums, return_counts=True)
    for n, c in zip(unique_n, counts_n):
        print(f"      含 {n} 个目标的图: {c} 张 ({c/len(sample_nums)*100:.1f}%)")

    # --- D. 宽高比分析 ---
    w, h = all_boxes[:, 3], all_boxes[:, 4]
    aspect_ratios = w[h > 0] / h[h > 0]   # 向量化，排除 h=0 的填充位
    print(f"\n  【4. 标注框宽高比】")
    print(f"    均值: {aspect_ratios.mean():.2f}, 中位数: {np.median(aspect_ratios):.2f}")
    print(f"    标准差: {aspect_ratios.std():.2f}")
    print(f"    范围: [{aspect_ratios.min():.2f}, {aspect_ratios.max():.2f}]")

    # --- E. 相似类别 ---
    print(f"\n  【5. 类别相似性分析】")
    print("    - 变形(0)、位移(2)、反装(3) 均为紧固件的位置/形态异常，")
    print("      视觉特征可能相似，易造成类间混淆。")
    print("    - 缺失(1) 与其他类别差异较大（紧固件不存在），相对易于区分。")
    print("    - 断裂(5) 可能有局部纹理变化，与变形(0) 存在一定混淆风险。")
    print("    建议: 关注混淆矩阵中 Defect-Defect 类间误差。")

    # --- F. 总结 ---
    print(f"\n  【总结】")
    print("    主要问题:")
    print("      1) 正常/缺陷类别极端不平衡（~6:1）")
    print("      2) 目标在 2021×2048 大图中占比较小，属小目标检测场景")
    print("      3) 部分缺陷类别视觉相似，需关注类间混淆")
    print("    建议改进方向:")
    print("      - 损失函数: Focal Loss 或 Class-balanced Loss")
    print("      - 数据增强: Mosaic、CutMix 及针对缺陷类的过采样")
    print("      - 输入策略: 切图 (tiling) 或高分辨率分支")
    print("=" * 65)


# ========================== 主流程 ==========================

def main():
    print("=" * 55)
    print("  任务 1: RFDD 数据准备与可视化")
    print("=" * 55)

    # ----- 1. 加载数据 -----
    images_tv, labels_tv, filenames_tv = load_trainval_h5()
    test_images, test_labels, test_fns = load_test_png()

    # 拆分 train&val
    train_idx, val_idx = split_trainval(labels_tv, filenames_tv)
    train_images = images_tv[train_idx]
    train_labels = labels_tv[train_idx]
    val_images = images_tv[val_idx]
    val_labels = labels_tv[val_idx]
    train_fns = filenames_tv[train_idx]
    val_fns = filenames_tv[val_idx]

    print(f"\n  数据加载完成！")
    print(f"    训练集:  图像 {len(train_images)} 张")
    print(f"    验证集:  图像 {len(val_images)} 张")
    print(f"    测试集:  图像 {len(test_images)} 张")

    # ----- 2. 标注框可视化（各数据集选若干张）-----
    print("\n" + "-" * 55)
    print("  标注框可视化")
    print("-" * 55)

    np.random.seed(42)
    visualize_samples(train_images, train_labels, train_fns, "Train", num_vis=5,
                      output_prefix="vis_boxes")
    visualize_samples(val_images, val_labels, val_fns, "Val", num_vis=5,
                      output_prefix="vis_boxes")
    visualize_samples(test_images, test_labels, test_fns, "Test", num_vis=5,
                      output_prefix="vis_boxes")

    # ----- 3. 类别统计 -----
    print("\n" + "-" * 55)
    print("  类别统计")
    print("-" * 55)

    train_cnt, train_total = count_classes(train_labels, "Train")
    val_cnt, val_total = count_classes(val_labels, "Val")
    test_cnt, test_total = count_classes(test_labels, "Test")

    print_class_table(train_cnt, val_cnt, test_cnt, train_total, val_total, test_total)
    plot_class_distribution(train_cnt, val_cnt, test_cnt)

    # ----- 4. 数据集问题分析 -----
    analyze_dataset(train_labels, val_labels, test_labels, labels_tv)

    # ----- 完成后 -----
    print(f"\n所有输出文件已保存至: {OUTPUT_DIR.resolve()}")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
