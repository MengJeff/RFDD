"""
任务 2：YOLOv8 模型训练与评估
================================
模型:   YOLOv8s (11.2M 参数)
GPU:    NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
输入:   640×640 (从 2021×2048 自动缩放)
Batch:  8 或 16 (根据显存调整)
Epoch:  100
学习率: lr0=0.001, cosine schedule
优化器: AdamW

前置:  需先运行 数据集分类.py 创建 YOLO 格式数据集
流程:
  1. GPU / 环境检查
  2. 训练 YOLOv8s
  3. 在测试集上评估，输出指标和检测结果图
  4. 统计推理速度与参数量
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ======================== 路径配置 ========================
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "Data_Prep"
OUTPUT_DIR = BASE_DIR / "outputs_task2_yolov8s"
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# YOLO 格式输出目录 (由数据集分类.py 创建)
YOLO_ROOT = WORK_DIR / "RFDD_Grouping"
YOLO_IMAGES_TRAIN = YOLO_ROOT / "images" / "train"
YOLO_IMAGES_VAL = YOLO_ROOT / "images" / "val"
YOLO_IMAGES_TEST = YOLO_ROOT / "images" / "test"
YOLO_LABELS_TRAIN = YOLO_ROOT / "labels" / "train"
YOLO_LABELS_VAL = YOLO_ROOT / "labels" / "val"
YOLO_LABELS_TEST = YOLO_ROOT / "labels" / "test"

CLASS_NAMES_EN = ["Deformed", "Missing", "Displaced",
                   "Inverted", "Normal", "Fractured"]
NUM_CLASSES = 6

# 超参数
CONFIG = {
    "model": "yolov8s.pt",      # 选取模型名称YOLOv8 small
    "imgsz": 640,               # 输入尺寸，训练时会自动缩放原图到640x640
    "batch": 8,                 # 批量大小，受GPU显存限制，RTX 4060 Laptop 8GB通常可用8或16
    "epochs": 100,              # 迭代次数
    "lr0": 0.001,               # 学习率
    "patience": 20,             # 早停轮数，若验证指标不提升超过20轮则停止训练
    "workers": 1,               # 数据加载器的工作进程数，受内存限制
    "device": 0,                # GPU 0，选用电脑第一个GPU，如果没有GPU则自动使用CPU
    "project": str(OUTPUT_DIR / "runs_s"), # 训练结果保存目录
    "name": "yolov8s_rfdd",               # 训练结果子目录名称
}

RESULT_DIR = BASE_DIR / "outputs_task2_yolov8s"  # 评估结果保存目录，包含测试集指标和检测结果图
RESULT_DIR.mkdir(exist_ok=True)   #若目录中有文件，会自动覆盖，不会报错


# ===================== 1. 环境检查 =====================

def check_environment():
    """检查 PyTorch + CUDA + ultralytics 环境"""
    print("=" * 60)
    print("  环境检查")
    print("=" * 60)

    import torch
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 名称:     {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存:         {mem_gb:.1f} GB")
        print(f"  CUDA 版本:    {torch.version.cuda}")

    try:
        import ultralytics
        print(f"  ultralytics:  {ultralytics.__version__}")
    except ImportError:
        print("  [错误] 未安装 ultralytics，请执行: conda install ultralytics")
        sys.exit(1)

    print("-" * 60)


# ===================== 2. YOLOv8 训练 =====================

def train_yolo(yaml_path):
    """训练 YOLOv8s"""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  YOLOv8s 训练开始")
    print("=" * 60)
    print(f"  模型:      {CONFIG['model']}")
    print(f"  输入尺寸:  {CONFIG['imgsz']}×{CONFIG['imgsz']}")
    print(f"  Batch:     {CONFIG['batch']}")
    print(f"  Epochs:    {CONFIG['epochs']}")
    print(f"  学习率:    {CONFIG['lr0']} (cosine schedule)")
    print(f"  早停:      patience={CONFIG['patience']}")
    print(f"  设备:      GPU (cuda:{CONFIG['device']})")
    print("-" * 60)

    model = YOLO(CONFIG['model'])
    results = model.train(
        data=str(yaml_path),
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        epochs=CONFIG['epochs'],
        lr0=CONFIG['lr0'],
        patience=CONFIG['patience'],
        workers=CONFIG['workers'],
        device=CONFIG['device'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=True,
        verbose=True,
    )
    print("\n  训练完成!")
    return model, results


# ===================== 3. 测试集评估 =====================

def evaluate_test(model, yaml_path):
    """在测试集上评估：mAP / 检测结果图 / 训练曲线"""
    print("\n" + "=" * 60)
    print("  测试集评估")
    print("=" * 60)

    # ----- 3a. 计算 mAP 等指标 -----
    metrics = model.val(
        data=str(yaml_path),
        split='test',
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        workers=CONFIG['workers'],
        device=CONFIG['device'],
        project=CONFIG['project'],
        name=CONFIG['name'] + '_test_eval',
        exist_ok=True,
    )

    print(f"\n  {'='*45}")
    print(f"  测试集指标汇总")
    print(f"  {'='*45}")
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision (P): {metrics.box.mp:.4f}")
    print(f"  Recall (R):    {metrics.box.mr:.4f}")

    # 每类别 AP@0.5
    if hasattr(metrics.box, 'ap') and len(metrics.box.ap) > 0:
        print(f"\n  各类别 AP@0.5:")
        for i, ap in enumerate(metrics.box.ap):
            if i < NUM_CLASSES:
                print(f"    {CLASS_NAMES_EN[i]:>12s}: {ap:.4f}")

    # 完整指标字典
    try:
        rd = metrics.results_dict
        print(f"\n  详细指标:")
        for k, v in rd.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v:.4f}")
            elif isinstance(v, (list, np.ndarray)):
                print(f"    {k}: {np.round(v, 4)}")
            else:
                print(f"    {k}: {v}")
    except Exception:
        pass

    # ----- 3b. 预测并保存检测结果图 -----
    print(f"\n  预测测试集并保存检测结果...")
    results_pred = model.predict(
        source=str(YOLO_IMAGES_TEST),
        imgsz=CONFIG['imgsz'],
        conf=0.25,
        iou=0.45,
        device=CONFIG['device'],
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(RESULT_DIR),
        name='predictions',
        exist_ok=True,
    )
    print(f"  检测结果: {RESULT_DIR / 'predictions'}")

    # ----- 3c. 绘制训练曲线 -----
    plot_training_curves()

    return metrics


def find_results_csv():
    """查找训练产生的 results.csv"""
    run_dir = Path(CONFIG['project']) / CONFIG['name']
    if (run_dir / "results.csv").exists():
        return run_dir / "results.csv"
    # ultralytics 可能放在 runs/detect/ 下
    alt_dir = Path(CONFIG['project']) / "detect" / CONFIG['name']
    if (alt_dir / "results.csv").exists():
        return alt_dir / "results.csv"
    return None


def parse_csv_to_dict(csv_path):
    """用标准库 csv 读取 results.csv，返回 {列名: [值列表]} 的字典"""
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for key, val in row.items():
                if key.strip() == '':
                    continue
                data.setdefault(key.strip(), []).append(float(val.strip()))
    return data


def plot_training_curves():
    """从 results.csv 绘制 loss / mAP / P / R 曲线"""
    csv_path = find_results_csv()
    if csv_path is None:
        print("  [警告] 未找到 results.csv，跳过曲线绘制。")
        return

    data = parse_csv_to_dict(csv_path)
    columns = list(data.keys())
    print(f"\n  训练曲线数据: {csv_path}")

    loss_cols = [c for c in columns if 'loss' in c.lower()]
    metric_cols = [c for c in columns
                   if any(k in c.lower() for k in ['map', 'precision', 'recall'])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col in loss_cols:
        axes[0].plot(data[col], label=col, linewidth=1)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for col in metric_cols:
        axes[1].plot(data[col], label=col, linewidth=1)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Metric')
    axes[1].set_title('Validation Metrics'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle("YOLOv8s Training Curves — RFDD", fontsize=14)
    plt.tight_layout()
    save_path = RESULT_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  曲线图已保存: {save_path}")

    # 最终轮指标
    if metric_cols and all(col in data for col in metric_cols):
        n_epochs = len(data[metric_cols[0]])
        print(f"  最终轮 (epoch {n_epochs}) 指标:")
        for col in metric_cols:
            if col in data and len(data[col]) > 0:
                print(f"    {col}: {data[col][-1]:.4f}")


# ===================== 4. 推理速度 & 参数量 =====================

def benchmark_speed(model):
    """测量参数量和 GPU 推理时间"""
    import torch, time

    print("\n" + "=" * 60)
    print("  推理速度 & 参数量")
    print("=" * 60)

    total_p = sum(p.numel() for p in model.model.parameters())
    trainable_p = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  参数量:       {total_p / 1e6:.2f} M (总)")
    print(f"  可训练:       {trainable_p / 1e6:.2f} M")
    print(f"  模型大小:     ~{total_p * 4 / 1024**2:.1f} MB (fp32)")

    device = next(model.model.parameters()).device
    dummy = torch.randn(1, 3, CONFIG['imgsz'], CONFIG['imgsz']).to(device)

    for _ in range(10):
        _ = model.model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    n_runs = 100
    for _ in range(n_runs):
        _ = model.model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    avg_ms = elapsed / n_runs * 1000
    fps = 1000 / avg_ms
    print(f"  推理时间:     {avg_ms:.2f} ms/张")
    print(f"  FPS:          {fps:.1f}")


# ===================== 主流程 =====================

def main():
    print("=" * 60)
    print("  任务 2: YOLOv8 — 铁路扣件缺陷检测")
    print("=" * 60)

    # 1. 环境检查
    check_environment()

    # 2. 检查数据集 (应由数据集分类.py 创建)
    yaml_path = YOLO_ROOT / "RFDD.yaml"   # 关键路径配置，yolo模型通过.yaml文件加载数据集，运行前需要确保路径正确
    if not yaml_path.exists():
        print(f"  [错误] 未找到 {yaml_path}")
        print("  请先运行 数据集分类.py")
        sys.exit(1)
    if not YOLO_IMAGES_TRAIN.exists() or not any(YOLO_IMAGES_TRAIN.iterdir()):
        print(f"  [错误] 训练集目录为空: {YOLO_IMAGES_TRAIN}")
        print("  请先运行 数据集分类.py")
        sys.exit(1)

    # 统计各集图片数量
    n_train = len(list(YOLO_IMAGES_TRAIN.glob("*.png")))
    n_val = len(list(YOLO_IMAGES_VAL.glob("*.png")))
    n_test = len(list(YOLO_IMAGES_TEST.glob("*.png")))

    # 打印超参数
    print(f"\n  {'='*60}")
    print(f"  训练超参数总览")
    print(f"  {'='*60}")
    print(f"  YOLO 版本:    YOLOv8s (ultralytics)")
    print(f"  Backbone:     CSPDarknet (11.2M 参数)")
    print(f"  输入尺寸:     {CONFIG['imgsz']}×{CONFIG['imgsz']}")
    print(f"  Batch size:   {CONFIG['batch']}")
    print(f"  Epochs:       {CONFIG['epochs']}")
    print(f"  学习率:       lr0={CONFIG['lr0']} (cosine schedule)")
    print(f"  优化器:       AdamW")
    print(f"  早停:         patience={CONFIG['patience']}")
    print(f"  数据增强:     mosaic, hsv, flip, scale (ultralytics 默认)")
    print(f"  数据集配置:   {yaml_path}")
    print(f"  训练集:       {n_train} 张")
    print(f"  验证集:       {n_val} 张")
    print(f"  测试集:       {n_test} 张")
    print(f"  类别数:       {NUM_CLASSES}")
    print(f"  GPU:          RTX 4060 Laptop (8GB)")

    # 3. 训练模型 / 加载已有模型
    from ultralytics import YOLO
    best_pt = Path(CONFIG['project']) / CONFIG['name'] / 'weights' / 'best.pt'
    if best_pt.exists():
        print(f"\n  检测到已有权重: {best_pt}")
        print("  跳过训练，直接加载模型进行评估。")
        model = YOLO(str(best_pt))
    else:
        model, _ = train_yolo(yaml_path)

    # 4. 测试集评估
    evaluate_test(model, yaml_path)

    # 5. 推理速度
    benchmark_speed(model)

    print("\n" + "=" * 60)
    print("  任务 2 完成!")
    print(f"  训练权重:  {CONFIG['project']}/{CONFIG['name']}/weights/")
    print(f"  评估结果:  {RESULT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
