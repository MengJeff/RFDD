"""
任务 3：Faster R-CNN 模型训练与评估
================================
模型:   Faster R-CNN (ResNet-50-FPN V2, 43.3M 参数)
GPU:    NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
输入:   640×640 (为公平起见，从 2021×2048 自动缩放)
Batch:  2 或 4 (根据显存调整)
Epoch:  100
学习率: lr0=0.001, cosine schedule
优化器: AdamW

前置:  需先运行 数据集分类.py 创建 YOLO 格式数据集
流程:
  1. GPU / 环境检查
  2. 训练 Faster R-CNN (ResNet-50-FPN V2)
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
OUTPUT_DIR = BASE_DIR / "outputs_task3_fasterrcnn_resnet50_fpn_v2"
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
    "model": "fasterrcnn_resnet50_fpn_v2",  # 选取模型名称Faster R-CNN ResNet-50-FPN V2
    "imgsz": 640,                            # 输入尺寸，训练时统一缩放到640×640
    "batch": 4,                              # 批量大小，Faster R-CNN显存占用较大，RTX 4060 Laptop 8GB 可用 2~4
    "epochs": 100,                           # 迭代次数
    "lr": 0.001,                             # 学习率
    "patience": 20,                          # 早停轮数，若验证指标不提升超过20轮则停止训练
    "workers": 1,                            # 数据加载器的工作进程数，受内存限制
    "device": 0,                             # GPU 0，选用电脑第一个GPU，如果没有GPU则自动使用CPU
    "momentum": 0.9,                         # SGD 动量（仅 SGD 模式使用）
    "weight_decay": 0.0005,                  # 权重衰减
}

RESULT_DIR = BASE_DIR / "outputs_task3_fasterrcnn_resnet50_fpn_v2"  # 评估结果保存目录，包含测试集指标和检测结果图
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
    """Faster R-CNN 要求: (list_of_images, list_of_targets)"""
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


# ===================== 2. Faster R-CNN 训练 =====================

def get_model(num_classes):
    """构建 Faster R-CNN ResNet-50-FPN V2，替换分类头适配 RFDD 类别数"""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 = 背景
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """单轮训练，返回各 loss 分量均值"""
    import torch

    model.train()
    total_loss = 0.0
    loss_cls_sum = 0.0
    loss_box_sum = 0.0
    loss_obj_sum = 0.0
    loss_rpn_sum = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        loss_cls_sum += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
        loss_box_sum += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
        loss_obj_sum += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
        loss_rpn_sum += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()

        if (i + 1) % 20 == 0:
            print(f"  Epoch [{epoch:3d}] Batch [{i+1:3d}/{len(data_loader):3d}]  "
                  f"Loss: {losses.item():.4f}")

    n = len(data_loader)
    return (total_loss / n, loss_cls_sum / n, loss_box_sum / n,
            loss_obj_sum / n, loss_rpn_sum / n)


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


def train_fasterrcnn():
    """训练 Faster R-CNN ResNet-50-FPN V2"""
    import torch
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("  Faster R-CNN (ResNet-50-FPN V2) 训练开始")
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
        writer.writerow(["epoch", "train_loss", "train_loss_classifier",
                         "train_loss_box_reg", "train_loss_objectness",
                         "train_loss_rpn_box_reg", "val_mAP50", "lr"])

    best_map = 0.0
    patience_counter = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        t_loss, t_cls, t_box, t_obj, t_rpn = train_one_epoch(
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
            writer.writerow([epoch, t_loss, t_cls, t_box, t_obj, t_rpn,
                             val_map, current_lr])

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


def _match_class_predictions(pred_items, gt_boxes_dict, iou_thresh):
    """按置信度从高到低匹配单个类别的预测框与真实框。"""
    import torch
    from torchvision.ops import box_iou

    pred_items = sorted(pred_items, key=lambda x: x[2], reverse=True)
    tp = torch.zeros(len(pred_items), dtype=torch.float32)
    fp = torch.zeros(len(pred_items), dtype=torch.float32)
    scores = torch.tensor([item[2] for item in pred_items], dtype=torch.float32)
    gt_matched = {
        img_id: torch.zeros(len(boxes), dtype=torch.bool)
        for img_id, boxes in gt_boxes_dict.items()
    }

    for i, (img_id, box, _) in enumerate(pred_items):
        gt_boxes = gt_boxes_dict.get(img_id, torch.zeros(0, 4))
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
        max_iou, max_idx = ious.max(0)
        if max_iou >= iou_thresh and not gt_matched[img_id][max_idx]:
            tp[i] = 1
            gt_matched[img_id][max_idx] = True
        else:
            fp[i] = 1

    return scores, tp, fp


def plot_confusion_matrix(all_preds, all_gts, num_classes, conf_thresh=0.25, iou_thresh=0.5):
    """绘制混淆矩阵与归一化混淆矩阵，格式接近 Ultralytics 输出。"""
    import torch
    from torchvision.ops import box_iou

    size = num_classes + 1
    matrix = np.zeros((size, size), dtype=np.int64)
    bg = num_classes

    for pred, gt in zip(all_preds, all_gts):
        pred_keep = pred["scores"] >= conf_thresh
        pred_boxes = pred["boxes"][pred_keep]
        pred_labels = pred["labels"][pred_keep]
        pred_scores = pred["scores"][pred_keep]

        valid_pred = (pred_labels >= 1) & (pred_labels <= num_classes)
        pred_boxes = pred_boxes[valid_pred]
        pred_labels = pred_labels[valid_pred]
        pred_scores = pred_scores[valid_pred]

        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        valid_gt = (gt_labels >= 1) & (gt_labels <= num_classes)
        gt_boxes = gt_boxes[valid_gt]
        gt_labels = gt_labels[valid_gt]
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for p_box, p_label in zip(pred_boxes, pred_labels):
            pred_cls = int(p_label.item()) - 1
            if len(gt_boxes) == 0:
                matrix[bg, pred_cls] += 1
                continue

            ious = box_iou(p_box.unsqueeze(0), gt_boxes)[0]
            max_iou, max_idx = ious.max(0)
            if max_iou >= iou_thresh and not gt_matched[max_idx]:
                true_cls = int(gt_labels[max_idx].item()) - 1
                matrix[true_cls, pred_cls] += 1
                gt_matched[max_idx] = True
            else:
                matrix[bg, pred_cls] += 1

        for g_label, matched in zip(gt_labels, gt_matched):
            if not matched:
                true_cls = int(g_label.item()) - 1
                matrix[true_cls, bg] += 1

    labels = CLASS_NAMES_EN + ["Background"]
    for data, name, title, fmt in [
        (matrix, "confusion_matrix.png", "Confusion Matrix", "d"),
        (
            matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1),
            "confusion_matrix_normalized.png",
            "Normalized Confusion Matrix",
            ".2f",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(data, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        threshold = data.max() * 0.55 if data.size else 0
        for i in range(size):
            for j in range(size):
                value = data[i, j]
                text = format(value, fmt)
                color = "white" if value > threshold else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

        plt.tight_layout()
        save_path = RESULT_DIR / name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {save_path}")


def plot_pr_and_confidence_curves(all_preds, all_gts, num_classes, iou_thresh=0.5):
    """绘制 BoxPR/P/R/F1 曲线，文件命名仿照 Ultralytics。"""
    recall_grid = np.linspace(0, 1, 101)
    conf_grid = np.linspace(0, 1, 101)
    pr_interp = []
    p_conf_interp = []
    r_conf_interp = []
    f1_conf_interp = []

    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    fig_p, ax_p = plt.subplots(figsize=(8, 6))
    fig_r, ax_r = plt.subplots(figsize=(8, 6))
    fig_f1, ax_f1 = plt.subplots(figsize=(8, 6))

    for c in range(1, num_classes + 1):
        pred_items = []
        gt_boxes_dict = {}
        gt_total = 0

        for img_id, (pred, gt) in enumerate(zip(all_preds, all_gts)):
            pred_mask = pred["labels"] == c
            for box, score in zip(pred["boxes"][pred_mask], pred["scores"][pred_mask]):
                pred_items.append((img_id, box, float(score.item())))

            gt_mask = gt["labels"] == c
            gt_boxes = gt["boxes"][gt_mask]
            gt_boxes_dict[img_id] = gt_boxes
            gt_total += len(gt_boxes)

        scores, tp, fp = _match_class_predictions(pred_items, gt_boxes_dict, iou_thresh)
        name = CLASS_NAMES_EN[c - 1]

        if len(scores) == 0 or gt_total == 0:
            precision = np.array([0.0])
            recall = np.array([0.0])
            conf = np.array([1.0])
        else:
            tp_cum = np.cumsum(tp.numpy())
            fp_cum = np.cumsum(fp.numpy())
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            recall = tp_cum / max(gt_total, 1)
            conf = scores.numpy()

        ax_pr.plot(recall, precision, linewidth=1.5, label=name)
        pr_interp.append(np.interp(recall_grid, recall, precision, left=precision[0], right=0))

        order = np.argsort(conf)
        conf_sorted = conf[order]
        p_sorted = precision[order]
        r_sorted = recall[order]
        f1_sorted = 2 * p_sorted * r_sorted / np.maximum(p_sorted + r_sorted, 1e-9)

        p_curve = np.interp(conf_grid, conf_sorted, p_sorted, left=p_sorted[0], right=0)
        r_curve = np.interp(conf_grid, conf_sorted, r_sorted, left=r_sorted[0], right=0)
        f1_curve = np.interp(conf_grid, conf_sorted, f1_sorted, left=f1_sorted[0], right=0)

        p_conf_interp.append(p_curve)
        r_conf_interp.append(r_curve)
        f1_conf_interp.append(f1_curve)
        ax_p.plot(conf_grid, p_curve, linewidth=1.2, label=name)
        ax_r.plot(conf_grid, r_curve, linewidth=1.2, label=name)
        ax_f1.plot(conf_grid, f1_curve, linewidth=1.2, label=name)

    if pr_interp:
        ax_pr.plot(recall_grid, np.mean(pr_interp, axis=0), color="black", linewidth=3, label="all classes")
    if p_conf_interp:
        ax_p.plot(conf_grid, np.mean(p_conf_interp, axis=0), color="black", linewidth=3, label="all classes")
    if r_conf_interp:
        ax_r.plot(conf_grid, np.mean(r_conf_interp, axis=0), color="black", linewidth=3, label="all classes")
    if f1_conf_interp:
        ax_f1.plot(conf_grid, np.mean(f1_conf_interp, axis=0), color="black", linewidth=3, label="all classes")

    for ax, xlabel, ylabel, title in [
        (ax_pr, "Recall", "Precision", f"Precision-Recall Curve (IoU={iou_thresh:.2f})"),
        (ax_p, "Confidence", "Precision", "Precision-Confidence Curve"),
        (ax_r, "Confidence", "Recall", "Recall-Confidence Curve"),
        (ax_f1, "Confidence", "F1", "F1-Confidence Curve"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for fig, name in [
        (fig_pr, "BoxPR_curve.png"),
        (fig_p, "BoxP_curve.png"),
        (fig_r, "BoxR_curve.png"),
        (fig_f1, "BoxF1_curve.png"),
    ]:
        plt.figure(fig.number)
        plt.tight_layout()
        save_path = RESULT_DIR / name
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {save_path}")


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

    # ----- 混淆矩阵与 PR / P / R / F1 曲线 -----
    print(f"\n  绘制混淆矩阵与 PR 曲线...")
    plot_confusion_matrix(all_preds, all_gts, NUM_CLASSES)
    plot_pr_and_confidence_curves(all_preds, all_gts, NUM_CLASSES)

    # ----- 保存检测结果图 -----
    print(f"\n  保存检测结果图...")
    save_predictions_visual(test_dataset, all_preds)

    # ----- 绘制训练曲线 -----
    plot_training_curves()

    return mAP50


def save_predictions_visual(dataset, all_preds, imgsz=640):
    from PIL import Image, ImageDraw, ImageFont

    save_dir = RESULT_DIR / "predictions"
    lbl_dir = save_dir / "labels"
    save_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    # ultralytics 色板 (BGR 转 RGB)
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

            # 标签（与 ultralytics 逻辑一致：框外上方 / 框内左上角）
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

    plt.suptitle("Faster R-CNN (ResNet-50-FPN V2) Training Curves — RFDD", fontsize=14)
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
    print("  任务 3: Faster R-CNN — 铁路扣件缺陷检测")
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
    print(f"  模型架构:     Faster R-CNN (torchvision)")
    print(f"  Backbone:     ResNet-50-FPN V2 (~41M 参数)")
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

    # 3. 训练模型 / 加载已有模型
    best_path = RESULT_DIR / "best_model.pth"
    if best_path.exists():
        print(f"\n  检测到已有权重: {best_path}")
        print("  跳过训练，直接加载模型进行评估。")
        model = torch.load(best_path, map_location=device, weights_only=False)
    else:
        model = train_fasterrcnn()
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
