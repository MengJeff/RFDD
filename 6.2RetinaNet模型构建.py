"""
任务 3：RetinaNet 模型训练与评估
================================
模型:   RetinaNet (ResNet-50-FPN V2, 38.2M 参数)
GPU:    NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
输入:   640×640 (为公平起见，从 2021×2048 自动缩放)
Batch:  2 或 4（根据显存调整）
Epoch:  100
学习率: lr0=0.001, cosine schedule
优化器: AdamW

前置:  需先运行 数据集分类.py 创建 YOLO 格式数据集
流程:
  1. GPU / 环境检查
  2. 训练 RetinaNet (ResNet-50-FPN V2)
  3. 在测试集上评估，输出指标和检测结果图
  4. 统计推理速度与参数量
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# ======================== 路径配置 ========================
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "Data_Prep"
OUTPUT_DIR = BASE_DIR / "outputs_task3_retinanet_resnet50_fpn_v2"
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

CLASS_NAMES_EN = ["Deformed", "Fractured", "Missing",
                  "Inverted", "Normal", "Displaced"]
NUM_CLASSES = 6

# 超参数
CONFIG = {
    "model": "retinanet_resnet50_fpn_v2",  # 选取模型名称RetinaNet ResNet-50-FPN V2
    "imgsz": 640,                            # 输入尺寸，训练时统一缩放到640×640
    "batch": 4,                              # 批量大小，RetinaNet显存占用较大，RTX 4060 Laptop 8GB 可用 2~4
    "epochs": 100,                           # 迭代次数
    "lr": 0.001,                             # 学习率
    "patience": 20,                          # 早停轮数，若验证指标不提升超过20轮则停止训练
    "workers": 1,                            # 数据加载器的工作进程数，受内存限制
    "device": 0,                             # GPU 0，选用电脑第一个GPU，如果没有GPU则自动使用CPU
    "momentum": 0.9,                         # SGD 动量（仅 SGD 模式使用）
    "weight_decay": 0.0005,                  # 权重衰减
}

RESULT_DIR = BASE_DIR / "outputs_task3_retinanet_resnet50_fpn_v2"  # 评估结果保存目录，包含测试集指标和检测结果图
RESULT_DIR.mkdir(exist_ok=True)


# ===================== 0. 数据集定义 =====================

class RFDDDataset(Dataset):
    """读取 YOLO 格式数据集，转换为目标检测所需格式，支持数据增强"""
    def __init__(self, img_dir, label_dir, imgsz=640, augment=False):
        import random as _random
        from torchvision import transforms as T

        self.img_paths = sorted(img_dir.glob("*.png"))
        self.label_dir = label_dir
        self.imgsz = imgsz
        self.augment = augment

        if augment:
            self.transform = T.Compose([
                T.Resize((imgsz, imgsz)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((imgsz, imgsz)),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        import torch, random
        from PIL import Image
        from torchvision.transforms import functional as TF

        img = Image.open(self.img_paths[idx]).convert("RGB")

        # 先读取标签（归一化坐标），再决定是否翻转
        label_path = self.label_dir / (self.img_paths[idx].stem + ".txt")
        labels_data = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    labels_data.append([cls, cx, cy, w, h])

        # 水平翻转增强（YOLO 风格，50% 概率）
        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)
            for b in labels_data:
                b[1] = 1.0 - b[1]  # 翻转 cx

        img = self.transform(img)

        boxes = []
        labels = []
        for b in labels_data:
            cls, cx, cy, w, h = b
            # YOLO 归一化坐标 → 绝对坐标
            x1 = (cx - w / 2) * self.imgsz
            y1 = (cy - h / 2) * self.imgsz
            x2 = (cx + w / 2) * self.imgsz
            y2 = (cy + h / 2) * self.imgsz
            boxes.append([x1, y1, x2, y2])
            labels.append(cls + 1)  # 0 = 背景，1~6 = 类别

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) \
               if len(boxes) > 0 else torch.zeros(0, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    """RetinaNet 要求: (list_of_images, list_of_targets)"""
    return tuple(zip(*batch))


# ===================== 1. 环境检查 =====================

def check_environment():
    """检查 PyTorch + CUDA + torchvision 环境"""

    print("=" * 60)
    print("  环境检查")
    print("=" * 60)

    import torch
    import torchvision
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  torchvision:  {torchvision.__version__}")
    print(f"  CUDA 可用:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 名称:     {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存:         {mem_gb:.1f} GB")
        print(f"  CUDA 版本:    {torch.version.cuda}")

    print("-" * 60)


# ===================== 2. RetinaNet 训练 =====================

def get_model(num_classes):
    """构建 RetinaNet ResNet-50-FPN V2，保留 COCO 91 类头（torchvision 0.22 头部替换有 bug）"""
    from torchvision.models.detection import retinanet_resnet50_fpn_v2

    # 注：不替换分类头，直接使用 91 类输出；
    # 仅 1~6 类被 RFDD 标签激活，7~90 类始终为背景，不影响训练
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """单轮训练，返回各 loss 分量均值（RetinaNet 仅分类 + 回归两个 loss）"""
    import torch

    model.train()
    total_loss = 0.0
    loss_cls_sum = 0.0
    loss_box_sum = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        loss_cls_sum += loss_dict.get('classification', torch.tensor(0.0)).item()
        loss_box_sum += loss_dict.get('bbox_regression', torch.tensor(0.0)).item()

        if (i + 1) % 20 == 0:
            print(f"  Epoch [{epoch:3d}] Batch [{i+1:3d}/{len(data_loader):3d}]  "
                  f"Loss: {losses.item():.4f}")

    n = len(data_loader)
    return total_loss / n, loss_cls_sum / n, loss_box_sum / n


def validate(model, data_loader, device):
    """在验证集上计算 mAP@0.5 与各类别 AP"""
    import torch

    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            preds = model(images)

            for pred in preds:
                all_preds.append({k: v.cpu() for k, v in pred.items()})
            for t in targets:
                all_gts.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in t.items()})

    return compute_mAP(all_preds, all_gts, NUM_CLASSES)[0]  # 训练验证只需 mAP50


def train_retinanet():
    """训练 RetinaNet ResNet-50-FPN V2"""
    import torch
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("  RetinaNet (ResNet-50-FPN V2) 训练开始")
    print("=" * 60)
    print(f"  模型:      {CONFIG['model']}")
    print(f"  输入尺寸:  {CONFIG['imgsz']}×{CONFIG['imgsz']}")
    print(f"  Batch:     {CONFIG['batch']}")
    print(f"  Epochs:    {CONFIG['epochs']}")
    print(f"  学习率:    {CONFIG['lr']} (cosine schedule)")
    print(f"  优化器:    AdamW")
    print(f"  早停:      patience={CONFIG['patience']}")
    print(f"  设备:      GPU (cuda:{CONFIG['device']})")
    print("-" * 60)

    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    # 数据集
    train_dataset = RFDDDataset(YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN, CONFIG['imgsz'], augment=True)
    val_dataset = RFDDDataset(YOLO_IMAGES_VAL, YOLO_LABELS_VAL, CONFIG['imgsz'], augment=False)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch'], shuffle=True,
                              num_workers=CONFIG['workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch'], shuffle=False,
                            num_workers=CONFIG['workers'], collate_fn=collate_fn)
    print(f"  训练集: {len(train_dataset)} 张,  验证集: {len(val_dataset)} 张")

    # 模型、优化器、调度器
    model = get_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                  weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # 训练日志
    csv_path = RESULT_DIR / "training_log.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_loss_classification",
                         "train_loss_bbox_regression", "val_mAP50", "lr"])

    best_map = 0.0
    patience_counter = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        t_loss, t_cls, t_box = train_one_epoch(
            model, optimizer, train_loader, device, epoch)

        val_map = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch [{epoch:3d}/{CONFIG['epochs']}]  "
              f"Train Loss: {t_loss:.4f}  "
              f"Val mAP@0.5: {val_map:.4f}  "
              f"LR: {current_lr:.6f}")

        scheduler.step()

        # 写入 CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, t_loss, t_cls, t_box, val_map, current_lr])

        # 保存最佳模型 & 早停
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model, RESULT_DIR / "best_model.pth")
            print(f"  >>> 新最佳模型 (mAP@0.5: {best_map:.4f})，已保存")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n  早停触发 (patience={CONFIG['patience']})，训练结束。")
                break

    print(f"\n  训练完成! 最佳 Val mAP@0.5: {best_map:.4f}")
    return model


# ===================== 3. 测试集评估 =====================

def compute_ap_voc(recall, precision):
    """VOC 11-point 插值法计算单类 AP"""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p_at_r = precision[recall >= t]
        ap += p_at_r.max().item() if len(p_at_r) > 0 else 0.0
    return ap / 11.0


def compute_mAP(all_preds, all_gts, num_classes):
    """计算 mAP@0.5、mAP@0.5:0.95 与各类别 AP (VOC 11-point)"""
    import torch
    from torchvision.ops import box_iou

    iou_thresholds = np.arange(0.50, 1.00, 0.05)  # 0.50, 0.55, ..., 0.95
    aps_50 = {}
    aps_all = {}                                   # AP averaged over all IoUs

    for c in range(1, num_classes + 1):            # 类别 1~6（跳过背景）
        # 收集该类所有预测和 GT（与 IoU 阈值无关）
        pred_boxes_list = []
        pred_scores_list = []
        pred_img_ids = []
        gt_cls_counts = {}
        gt_boxes_dict = {}

        for img_id, (pred, gt) in enumerate(zip(all_preds, all_gts)):
            mask = pred['labels'] == c
            p_boxes = pred['boxes'][mask]
            p_scores = pred['scores'][mask]
            order = torch.argsort(p_scores, descending=True)
            for idx in order:
                pred_boxes_list.append(p_boxes[idx])
                pred_scores_list.append(p_scores[idx].item())
                pred_img_ids.append(img_id)

            gt_mask = gt['labels'] == c
            g_boxes = gt['boxes'][gt_mask]
            gt_cls_counts[img_id] = len(g_boxes)
            gt_boxes_dict[img_id] = g_boxes

        total_gt = sum(gt_cls_counts.values())

        if len(pred_boxes_list) == 0:
            aps_50[CLASS_NAMES_EN[c - 1]] = 1.0 if total_gt == 0 else 0.0
            aps_all[CLASS_NAMES_EN[c - 1]] = 1.0 if total_gt == 0 else 0.0
            continue

        pred_boxes_t = torch.stack(pred_boxes_list)
        ap_over_ious = []

        for iou_thresh in iou_thresholds:
            tp = torch.zeros(len(pred_boxes_t))
            fp = torch.zeros(len(pred_boxes_t))
            gt_matched = {img_id: torch.zeros(len(gt_boxes_dict[img_id]), dtype=torch.bool)
                          for img_id in gt_boxes_dict}

            for i, (pb, img_id) in enumerate(zip(pred_boxes_t, pred_img_ids)):
                gt_boxes_img = gt_boxes_dict.get(img_id, torch.zeros(0, 4))
                if len(gt_boxes_img) == 0:
                    fp[i] = 1
                    continue
                ious = box_iou(pb.unsqueeze(0), gt_boxes_img)[0]
                max_iou, max_idx = ious.max(0)
                if max_iou >= iou_thresh and not gt_matched[img_id][max_idx]:
                    tp[i] = 1
                    gt_matched[img_id][max_idx] = True
                else:
                    fp[i] = 1

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recall = tp_cum / total_gt if total_gt > 0 else torch.zeros_like(tp_cum)
            precision = tp_cum / (tp_cum + fp_cum + 1e-6)
            ap_over_ious.append(compute_ap_voc(recall, precision))

        aps_50[CLASS_NAMES_EN[c - 1]] = ap_over_ious[0]      # IoU=0.50
        aps_all[CLASS_NAMES_EN[c - 1]] = float(np.mean(ap_over_ious))

    mAP50 = float(np.mean(list(aps_50.values())))
    mAP50_95 = float(np.mean(list(aps_all.values())))
    return mAP50, mAP50_95, aps_50, aps_all


def compute_pr(all_preds, all_gts, num_classes, conf_thresh=0.25, iou_thresh=0.5):
    """计算总体 Precision / Recall 与各类别 P / R"""
    import torch
    from torchvision.ops import box_iou

    p_per_class = {}
    r_per_class = {}
    tp_all = 0
    fp_all = 0
    fn_all = 0

    for c in range(1, num_classes + 1):
        pred_boxes_list = []
        pred_scores_list = []
        pred_img_ids = []
        gt_total = 0
        gt_boxes_dict = {}

        for img_id, (pred, gt) in enumerate(zip(all_preds, all_gts)):
            mask = (pred['labels'] == c) & (pred['scores'] >= conf_thresh)
            p_boxes = pred['boxes'][mask]
            p_scores = pred['scores'][mask]
            for box, score in zip(p_boxes, p_scores):
                pred_boxes_list.append(box)
                pred_scores_list.append(score.item())
                pred_img_ids.append(img_id)

            gt_mask = gt['labels'] == c
            g_boxes = gt['boxes'][gt_mask]
            gt_total += len(g_boxes)
            gt_boxes_dict[img_id] = g_boxes

        tp = 0; fp = 0
        gt_matched = {img_id: torch.zeros(len(gt_boxes_dict[img_id]), dtype=torch.bool)
                      for img_id in gt_boxes_dict}

        for pb, img_id in zip(pred_boxes_list, pred_img_ids):
            gt_boxes_img = gt_boxes_dict.get(img_id, torch.zeros(0, 4))
            if len(gt_boxes_img) == 0:
                fp += 1
                continue
            ious = box_iou(pb.unsqueeze(0), gt_boxes_img)[0]
            max_iou, max_idx = ious.max(0)
            if max_iou >= iou_thresh and not gt_matched[img_id][max_idx]:
                tp += 1
                gt_matched[img_id][max_idx] = True
            else:
                fp += 1

        fn = gt_total - tp
        p_per_class[CLASS_NAMES_EN[c - 1]] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_per_class[CLASS_NAMES_EN[c - 1]] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tp_all += tp; fp_all += fp; fn_all += fn

    P = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    R = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    return P, R, p_per_class, r_per_class


def evaluate_test(model, device):
    """在测试集上评估：mAP / 检测结果图 / 训练曲线"""
    import torch
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("  测试集评估")
    print("=" * 60)

    model.eval()

    test_dataset = RFDDDataset(YOLO_IMAGES_TEST, YOLO_LABELS_TEST, CONFIG['imgsz'], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch'], shuffle=False,
                             num_workers=CONFIG['workers'], collate_fn=collate_fn)

    # ----- 收集测试集预测与 GT -----
    all_preds = []
    all_gts = []
    print(f"  推理中... ({len(test_dataset)} 张测试图)")
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            for pred in preds:
                all_preds.append({k: v.cpu() for k, v in pred.items()})
            for t in targets:
                all_gts.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in t.items()})

    mAP50, mAP50_95, aps_50, aps_all = compute_mAP(all_preds, all_gts, NUM_CLASSES)
    P, R, p_per_class, r_per_class = compute_pr(all_preds, all_gts, NUM_CLASSES)

    # ----- 总体指标 -----
    print(f"\n  {'─' * 45}")
    print(f"  总体指标")
    print(f"  {'─' * 45}")
    print(f"  mAP@0.5:       {mAP50:.4f}")
    print(f"  mAP@0.5:0.95:  {mAP50_95:.4f}")
    print(f"  Precision (P): {P:.4f}")
    print(f"  Recall (R):    {R:.4f}")

    # ----- 各类别 AP@0.5 与 AP@0.5:0.95 -----
    print(f"\n  {'─' * 45}")
    print(f"  各类别 AP@0.5 与 AP@0.5:0.95")
    print(f"  {'─' * 45}")
    print(f"  {'类别':<16s} {'AP@0.5':>8s}  {'AP@0.5:0.95':>10s}")
    print(f"  {'─' * 45}")
    for cid in range(NUM_CLASSES):
        name = CLASS_NAMES_EN[cid]
        print(f"  {name:<16s} {aps_50[name]:>8.4f}  {aps_all[name]:>10.4f}")

    # ----- 各类别 Precision / Recall -----
    print(f"\n  {'─' * 45}")
    print(f"  各类别 Precision / Recall")
    print(f"  {'─' * 45}")
    print(f"  {'类别':<16s} {'Precision':>10s}  {'Recall':>10s}")
    print(f"  {'─' * 45}")
    for cid in range(NUM_CLASSES):
        name = CLASS_NAMES_EN[cid]
        print(f"  {name:<16s} {p_per_class[name]:>10.4f}  {r_per_class[name]:>10.4f}")

    # ----- 保存检测结果图 -----
    print(f"\n  保存检测结果图...")
    save_predictions_visual(test_dataset, all_preds)

    # ----- 绘制训练曲线 -----
    plot_training_curves()

    return mAP50


def save_predictions_visual(dataset, all_preds, imgsz=640):
    """使用 PIL ImageDraw 渲染预测框，与 ultralytics 完全一致"""
    from PIL import Image, ImageDraw, ImageFont

    save_dir = RESULT_DIR / "predictions"
    lbl_dir = save_dir / "labels"
    save_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    # ultralytics 色板 (RGB)
    colors     = [(255, 42, 4), (235, 219, 11), (243, 243, 243),
                  (183, 223, 0), (104, 31, 17), (221, 111, 255)]
    txt_colors = [(255, 255, 255), (0, 0, 0), (0, 0, 0),
                  (0, 0, 0), (255, 255, 255), (0, 0, 0)]
    lw = max(round(sum([imgsz, imgsz]) / 2 * 0.003), 2)
    font_size = max(round(sum([imgsz, imgsz]) / 2 * 0.035), 12)

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for idx in range(len(dataset)):
        img_tensor, _ = dataset[idx]
        # tensor [C,H,W] → PIL Image
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil, "RGBA")

        pred = all_preds[idx]
        keep = pred['scores'] > 0.25
        label_lines = []

        for box, label, score in zip(pred['boxes'][keep], pred['labels'][keep],
                                      pred['scores'][keep]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cls_id = int(label.item()) - 1
            color = colors[cls_id % len(colors)]
            txt_c = txt_colors[cls_id % len(txt_colors)]

            # 边框
            draw.rectangle([x1, y1, x2, y2], width=lw, outline=color)

            # 标签
            if 0 <= cls_id < len(CLASS_NAMES_EN):
                text = f"{CLASS_NAMES_EN[cls_id]} {score:.2f}"
                tw, th = font.getbbox(text)[2:4] if hasattr(font, 'getbbox') else font.getsize(text)
                outside = y1 >= th
                tx = x1
                ty = y1 - th if outside else y1
                draw.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=color)
                draw.text((tx + 1, ty + 1), text, fill=txt_c, font=font)

            # YOLO 格式标签
            cx_n = (x1 + x2) / 2 / imgsz
            cy_n = (y1 + y2) / 2 / imgsz
            bw = (x2 - x1) / imgsz
            bh = (y2 - y1) / imgsz
            label_lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {bw:.6f} {bh:.6f} {score:.6f}")

        stem = Path(dataset.img_paths[idx]).stem
        img_pil.save(save_dir / f"{stem}.jpg", quality=95)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines), encoding='utf-8')

    print(f"  检测结果: {save_dir}")
    print(f"  预测标签: {lbl_dir}")


def plot_training_curves():
    """从 training_log.csv 绘制 loss / mAP 曲线"""
    csv_path = RESULT_DIR / "training_log.csv"
    if not csv_path.exists():
        print("  [警告] 未找到 training_log.csv，跳过曲线绘制。")
        return

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for key, val in row.items():
                data.setdefault(key.strip(), []).append(float(val.strip()))

    if not data:
        return

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    loss_cols = [c for c in data if 'loss' in c.lower()]
    for col in loss_cols:
        axes[0].plot(data[col], label=col, linewidth=1)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    # mAP 曲线
    if 'val_mAP50' in data:
        axes[1].plot(data['val_mAP50'], label='Val mAP@0.5', linewidth=1.5)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mAP@0.5')
    axes[1].set_title('Validation mAP'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle("RetinaNet (ResNet-50-FPN V2) Training Curves — RFDD", fontsize=14)
    plt.tight_layout()
    save_path = RESULT_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  曲线图已保存: {save_path}")

    # 最终轮指标
    if 'val_mAP50' in data and len(data['val_mAP50']) > 0:
        n_epochs = len(data['val_mAP50'])
        print(f"  最终轮 (epoch {n_epochs}) Val mAP@0.5: {data['val_mAP50'][-1]:.4f}")
    if 'train_loss' in data and len(data['train_loss']) > 0:
        print(f"  最终轮 (epoch {len(data['train_loss'])}) Train Loss: {data['train_loss'][-1]:.4f}")


# ===================== 4. 推理速度 & 参数量 =====================

def benchmark_speed(model):
    """测量参数量和 GPU 推理时间"""
    import torch
    import time

    print("\n" + "=" * 60)
    print("  推理速度 & 参数量")
    print("=" * 60)

    model.eval()

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量:       {total_p / 1e6:.2f} M (总)")
    print(f"  可训练:       {trainable_p / 1e6:.2f} M")
    print(f"  模型大小:     ~{total_p * 4 / 1024**2:.1f} MB (fp32)")

    device = next(model.parameters()).device
    dummy = [torch.randn(3, CONFIG['imgsz'], CONFIG['imgsz']).to(device)]

    # 预热
    for _ in range(10):
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    n_runs = 100
    for _ in range(n_runs):
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    avg_ms = elapsed / n_runs * 1000
    fps = 1000 / avg_ms
    print(f"  推理时间:     {avg_ms:.2f} ms/张")
    print(f"  FPS:          {fps:.1f}")


# ===================== 5. 日志模块 =====================

def setup_logger():
    """将后续所有 print 输出同时写入日志文件（UTF-8 编码）"""
    import sys
    from datetime import datetime

    RESULT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULT_DIR / f"log_{CONFIG['model']}_{timestamp}.txt"

    class Tee:
        def __init__(self, f1, f2):
            self.f1 = f1; self.f2 = f2
        def write(self, data):
            self.f1.write(data); self.f2.write(data)
        def flush(self):
            self.f1.flush(); self.f2.flush()

    sys.stdout = Tee(sys.stdout, open(str(log_path), 'w', encoding='utf-8', buffering=1))
    print(f"  日志文件: {log_path}\n")


# ===================== 主流程 =====================

def main():
    setup_logger()

    import torch

    print("=" * 60)
    print("  任务 3: RetinaNet — 铁路扣件缺陷检测")
    print("=" * 60)

    # 1. 环境检查
    check_environment()

    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')

    # 2. 检查数据集（应由数据集分类.py 创建）
    if not YOLO_IMAGES_TRAIN.exists() or not any(YOLO_IMAGES_TRAIN.iterdir()):
        print(f"  [错误] 训练集目录为空: {YOLO_IMAGES_TRAIN}")
        print("  请先运行 数据集分类.py")
        sys.exit(1)
    if not YOLO_IMAGES_VAL.exists() or not any(YOLO_IMAGES_VAL.iterdir()):
        print(f"  [错误] 验证集目录为空: {YOLO_IMAGES_VAL}")
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
    print(f"  模型架构:     RetinaNet (torchvision)")
    print(f"  Backbone:     ResNet-50-FPN V2 (~38M 参数)")
    print(f"  输入尺寸:     {CONFIG['imgsz']}×{CONFIG['imgsz']}")
    print(f"  Batch size:   {CONFIG['batch']}")
    print(f"  Epochs:       {CONFIG['epochs']}")
    print(f"  学习率:       lr0={CONFIG['lr']} (cosine schedule)")
    print(f"  优化器:       AdamW")
    print(f"  早停:         patience={CONFIG['patience']}")
    print(f"  训练集:       {n_train} 张")
    print(f"  验证集:       {n_val} 张")
    print(f"  测试集:       {n_test} 张")
    print(f"  类别数:       {NUM_CLASSES}")
    print(f"  GPU:          RTX 4060 Laptop (8GB)")

    # 3. 训练 / 加载已有模型
    best_path = RESULT_DIR / "best_model.pth"
    if best_path.exists():
        print(f"\n  检测到已有权重: {best_path}")
        print("  跳过训练，直接加载模型进行评估。")
        model = torch.load(best_path, map_location=device, weights_only=False)
    else:
        model = train_retinanet()
        if best_path.exists():
            model = torch.load(best_path, map_location=device, weights_only=False)
            print(f"\n  已加载最佳模型: {best_path}")
    model.to(device)

    # 4. 测试集评估
    evaluate_test(model, device)

    # 5. 推理速度
    benchmark_speed(model)

    print("\n" + "=" * 60)
    print("  任务 3 完成!")
    print(f"  最佳权重:  {RESULT_DIR / 'best_model.pth'}")
    print(f"  训练日志:  {RESULT_DIR / 'training_log.csv'}")
    print(f"  评估结果:  {RESULT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
