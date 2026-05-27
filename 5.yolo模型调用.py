"""
YOLOv8 训练模型调用 — 测试集评估与可视化
=============================================
加载训练好的 best.pt，对 RFDD_datasets/test 进行:
  1. 指标评估 (mAP / Precision / Recall / 各类别 AP)
  2. 预测结果可视化 (所有检测结果图)
  3. 推理速度统计
"""

import sys
from pathlib import Path
import yaml
import shutil
from PIL import Image

# ======================== 路径配置 ========================
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "RFDD_datasets"
WORK_DIR = BASE_DIR / "Data_Prep"

# ========== 全局模型变量（切换模型时只需修改此处） ==========
MODEL_NAME = "yolov8s"      # 模型名称，如 yolov8s / yolov8m / yolov8l
MODEL_SIZE = MODEL_NAME[-1]  # 模型尺寸后缀，如 s / m / l
# ===========================================================

# 训练好的模型权重
BEST_PT = BASE_DIR / f"outputs_task2_{MODEL_NAME}" / f"runs_{MODEL_SIZE}" / f"{MODEL_NAME}_rfdd" / "weights" / "best.pt"  # 模型权重路径，需与训练脚本保持一致

# 测试数据原始路径
TEST_IMG_SRC = DATASETS_DIR / "test" / "images"
TEST_LBL_SRC = DATASETS_DIR / "test" / "labels"


# YOLO 格式测试集（复用训练脚本创建的目录）
YOLO_TEST_IMAGES = WORK_DIR / "RFDD_Grouping" / "images" / "test"
YOLO_TEST_LABELS = WORK_DIR / "RFDD_Grouping" / "labels" / "test"

# 输出目录
RESULT_DIR = BASE_DIR / f"outputs_task2_{MODEL_NAME}"  # 评估结果保存目录，包含测试集指标和检测结果图
RESULT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    0: "Deformed", 1: "Fractured", 2: "Missing",
    3: "Inverted", 4: "Normal", 5: "Displaced",
}
# 推理参数
CONFIG = {
    "imgsz": 640,  #图像尺寸640x640，需与训练时保持一致
    "batch": 8,    #批量大小，评估时可适当增大以提升速度（根据GPU显存调整）推荐16
    "conf": 0.25,  #置信度阈值，默认0.25，评估时可适当调整以观察指标变化
    "iou": 0.45,   #NMS IoU阈值，默认0.45，评估时可适当调整以观察指标变化
    "device": 0,   #GPU 0，选用电脑第一个GPU，如果没有GPU则自动使用CPU
}


# ===================== 0. 检查文件 =====================

def check_files():
    """确认模型权重存在"""
    if not BEST_PT.exists():
        print(f"[错误] 未找到模型权重: {BEST_PT}")
        print("请先运行 yolo模型构建.py 完成训练。")
        sys.exit(1)
    print(f"  模型权重: {BEST_PT}")


# ===================== 1. 准备测试数据 =====================

def prepare_test_data():
    """确保 YOLO 格式的测试集存在（不存在则创建）"""
    # 如果训练脚本已创建过，直接复用
    if YOLO_TEST_IMAGES.exists() and any(YOLO_TEST_IMAGES.iterdir()):
        print(f"  复用已有 YOLO 测试集: {YOLO_TEST_IMAGES}")
        return

    print("  创建 YOLO 格式测试集...")
    YOLO_TEST_IMAGES.mkdir(parents=True, exist_ok=True)
    YOLO_TEST_LABELS.mkdir(parents=True, exist_ok=True)

    count = 0
    for png_file in sorted(TEST_IMG_SRC.glob("*.png")):
        count += 1

        # 图像: 统一转为 RGB 三通道，保留原始文件名
        img = Image.open(png_file).convert("RGB")
        img.save(YOLO_TEST_IMAGES / png_file.name)

        # 标签: 直接复制，保留原始文件名
        lbl_src = TEST_LBL_SRC / (png_file.stem + ".txt")
        lbl_dst = YOLO_TEST_LABELS / (png_file.stem + ".txt")
        if lbl_src.exists():
            shutil.copy(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("")

    print(f"  测试集准备完成: {count} 张 → {YOLO_TEST_IMAGES}")


def create_test_yaml():
    """为测试集单独创建 dataset.yaml（用于 model.val）"""
    yaml_path = WORK_DIR / "RFDD_Grouping" / "RFDD_test.yaml"
    # ultralytics 强制要求 train / val 字段存在，测试时指向 test 目录即可
    data = {
        'path': str((WORK_DIR / "RFDD_Grouping").resolve()),
        'train': 'images/test',
        'val': 'images/test',
        'test': 'images/test',
        'nc': 6,
        'names': CLASS_NAMES,
    }
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True),
                         encoding='utf-8')
    return yaml_path


# ===================== 2. 模型加载 =====================

def load_model():
    """加载训练好的模型"""
    from ultralytics import YOLO

    print(f"\n  加载模型: {BEST_PT}")
    model = YOLO(str(BEST_PT))
    return model


# ===================== 3. 指标评估 =====================

def evaluate_metrics(model, yaml_path):
    """在测试集上计算 mAP / Precision / Recall / 各类别 AP"""
    print("\n" + "=" * 55)
    print("  测试集指标评估")
    print("=" * 55)

    metrics = model.val(
        data=str(yaml_path),
        split='test',
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        conf=CONFIG['conf'],
        iou=CONFIG['iou'],
        device=CONFIG['device'],
        project=str(RESULT_DIR),
        name='metrics',
        exist_ok=True,
    )

    # ===== 汇总指标 =====
    print(f"\n  {'─' * 45}")
    print(f"  总体指标")
    print(f"  {'─' * 45}")
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision (P): {metrics.box.mp:.4f}")
    print(f"  Recall (R):    {metrics.box.mr:.4f}")

    # ===== 各类别 AP =====
    print(f"\n  {'─' * 45}")
    print(f"  各类别 AP@0.5 与 AP@0.5:0.95")
    print(f"  {'─' * 45}")
    print(f"  {'类别':<16s} {'AP@0.5':>8s}  {'AP@0.5:.95':>10s}")
    print(f"  {'─' * 45}")

    ap50_list = getattr(metrics.box, 'ap50', None)  # per-class AP@0.5
    ap_list = getattr(metrics.box, 'ap', None)       # per-class AP@0.5:0.95

    for cid in range(6):
        name = CLASS_NAMES[cid]
        val50 = ap50_list[cid] if ap50_list is not None and len(ap50_list) > cid else 0.0
        val95 = ap_list[cid] if ap_list is not None and len(ap_list) > cid else 0.0
        print(f"  {name:<16s} {val50:>8.4f}  {val95:>10.4f}")

    # ===== 各类别 Precision / Recall =====
    print(f"\n  {'─' * 45}")
    print(f"  各类别 Precision / Recall")
    print(f"  {'─' * 45}")
    print(f"  {'类别':<16s} {'Precision':>10s}  {'Recall':>10s}")
    print(f"  {'─' * 45}")

    p_list = getattr(metrics.box, 'p', None)
    r_list = getattr(metrics.box, 'r', None)

    for cid in range(6):
        name = CLASS_NAMES[cid]
        pv = p_list[cid] if p_list is not None and len(p_list) > cid else 0.0
        rv = r_list[cid] if r_list is not None and len(r_list) > cid else 0.0
        print(f"  {name:<16s} {pv:>10.4f}  {rv:>10.4f}")

    return metrics


# ===================== 4. 预测可视化 =====================

def run_prediction(model):
    """对所有测试图像进行预测，保存检测结果图"""
    print("\n" + "=" * 55)
    print("  预测可视化")
    print("=" * 55)

    results = model.predict(
        source=str(YOLO_TEST_IMAGES),
        imgsz=CONFIG['imgsz'],
        conf=CONFIG['conf'],
        iou=CONFIG['iou'],
        device=CONFIG['device'],
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(RESULT_DIR),
        name='predictions',
        exist_ok=True,
    )

    pred_dir = RESULT_DIR / "predictions"
    print(f"  检测结果图: {pred_dir}")
    return results


# ===================== 5. 推理速度 =====================

def benchmark_speed(model):
    """GPU 推理速度统计"""
    import torch, time

    print("\n" + "=" * 55)
    print("  推理速度")
    print("=" * 55)

    # 参数量
    total_p = sum(p.numel() for p in model.model.parameters())
    trainable_p = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  参数量:     {total_p / 1e6:.2f} M (总)")
    print(f"  可训练:     {trainable_p / 1e6:.2f} M")
    print(f"  模型大小:   ~{total_p * 4 / 1024**2:.1f} MB (fp32)")

    # GPU 推理耗时
    device = next(model.model.parameters()).device
    dummy = torch.randn(1, 3, CONFIG['imgsz'], CONFIG['imgsz']).to(device)

    for _ in range(10):
        _ = model.model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        _ = model.model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    avg_ms = elapsed / 100 * 1000
    fps = 1000 / avg_ms
    print(f"  推理时间:   {avg_ms:.2f} ms/张")
    print(f"  FPS:        {fps:.1f}")


# ===================== 主流程 =====================

def main():
    print("=" * 55)
    print(f"  {MODEL_NAME} 训练模型调用 — 测试集评估")
    print("=" * 55)

    # 0. 检查
    check_files()

    # 1. 准备测试数据
    prepare_test_data()
    yaml_path = create_test_yaml()
    print(f"  数据集配置: {yaml_path}")

    # 2. 加载模型
    model = load_model()

    # 3. 指标评估
    evaluate_metrics(model, yaml_path)

    # 4. 预测 & 可视化
    run_prediction(model)

    # 5. 推理速度
    benchmark_speed(model)

    print("\n" + "=" * 55)
    print("  测试集评估完成!")
    print(f"  指标结果: {RESULT_DIR / 'metrics'}")
    print(f"  检测结果: {RESULT_DIR / 'predictions'}")
    print("=" * 55)


if __name__ == "__main__":
    main()
