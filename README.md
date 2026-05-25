# RFDD — 铁路扣件缺陷检测 (Railway Fastener Defect Detection)

基于 YOLOv8 的铁路扣件状态检测系统，使用 RFDD（Railway Fastener Defect Dataset）数据集训练和评估目标检测模型，识别扣件的六种状态：变形、缺失、位移、反装、正常、断裂。

# 克隆项目需注意

因为GitHub文件上传大小限制，目前没有给文件夹中添加文件，文件下.txt或者.png文件没有内容，在克隆到本地时请删去。只要把原始数据文件替换为解压的文件，程序就可以运行，没有的文件夹也会自动生成，参考模型训练时间在1.2h左右

# 一些问题

1. .cache文件为训练模型时yolo自动生成的缓存文件，相当于索引使得文件读取更加快速。当重新训练模型时，需要将此缓存文件删掉。
2. 模型训练时，该ultralytics库会自动求解最优参数，使得先前输入的超参数被覆盖。对应参数“optimizer (str): 选择优化器。可选值包括 'SGD', 'Adam', 'AdamW', 'RMSProp' 等，或 'auto'（由系统自动选择为AdamW）”。系统最终选择的优化器为 AdamW，初始学习率为 0.001，动量为 0.9。同时，它将模型参数分为 3 个组（无衰减权重、有衰减权重、偏置值）分别进行梯度更新。
3. 在自定义 YOLO 的输入分辨率时，必须遵守一个基本原则：输入图像的宽度和高度必须是 32 的整数倍目前640*640属于中等分辨率，可能会丢失信息，增大该超参数，显存会增加，请谨慎考虑该参数与batch参数的配合，避免爆显存。
4. 针对目前问题“Normal 类样本远多于缺陷类样本”，会导致以下问题：由于“Normal”样本数量庞大，其产生的损失值（Loss）会主导梯度的更新方向。模型倾向于简单地将所有区域预测为“Normal”以快速降低整体损失，从而抑制了对稀有缺陷特征的学习。解决方法很多，目前在考虑“开启或调整Focal Loss（在超参数配置文件中设置 fl_gamma，例如设为 fl_gamma=1.5 或 2.0，Focal Loss通过降低易分类样本（如大量的Normal）的权重，使模型专注于难分类的样本。”

# 对于 `3.数据集分类.py` 程序运行结果的说明

## 标签名 `1-L1-WZ-GX3-R-YW-L1-1` 的含义

该名称是 RFDD 数据集内建的**结构化命名规则**，来源于 H5 文件中存储的原始文件名。用 `-` 分割后共 8 个字段：

| 字段 | 值 | 含义 |
|:---:|:---:|:---|
| 1 | `1` | 样本序号（第 1 号样本） |
| 2 | `L1` | 线路编号（第1线路，即 **L**ine **1**） |
| 3 | `WZ` | 站段代码（某铁路站段的拼音缩写） |
| 4 | `GX3` | 区段/子线路代码 |
| 5 | `R` | 左右轨标识：**R** = 右轨（Right rail） |
| 6 | `YW` | 缺陷类型代码：**YW** = 位移（对应 class_id=2, Displaced） |
| 7 | `L1` | 扣件位号：**L1** = 该轨上第1号扣件位置 |
| 8 | `1` | 样本序号（与字段1重复，起冗余校验作用） |

## 各缺陷代码对照表

| 代码 | class_id | 英文 | 中文 |
|:---:|:---:|:---|:---|
| `BX` | 0 | Deformed | 变形 |
| `DL` | 1 | Missing | 缺失 |
| `DS` / `YW` | 2 | Displaced | 位移 |
| `FZ` | 3 | Inverted | 反装 |
| `N` | 4 | Normal | 正常 |
| `CS` | 5 | Fractured | 断裂 |

---

## 项目结构

```
RFDD/
├── 检查python环境.py                   # 环境检查：PyTorch/CUDA/ultralytics 版本
├── 1.数据集预处理.py                    # 数据集去重（MD5 哈希）
├── 2.图片读取.py                        # 任务1：数据加载、标注可视化、统计分析
├── 3.数据集分类.py                      # 数据集划分与 YOLO 格式导出
├── 4.1yolo模型构建YOLOv8_small.py      # 任务2：YOLOv8s 训练与评估
├── 4.2yolo模型构建YOLOv8_m.py           # 任务2：YOLOv8m 训练与评估
├── 5.yolo模型调用.py                    # 已训练模型推理与评估
├── 6.测试集可视化.py                    # 测试集导入 YOLO 目录工具
├── 要求.txt                             # 作业要求说明
├── RFDD_datasets/                       # 原始数据集
│   ├── train&val/                       #   - 训练+验证集（HDF5 格式）
│   └── test/                            #   - 测试集（PNG + YOLO TXT）
├── Data_Prep/                           # 预处理后数据
│   ├── RFDD_clean/                      #   - 去重后的 H5 数据
│   └── RFDD_Grouping/                   #   - YOLO 格式数据集（train/val/test）
│       ├── RFDD.yaml                    #     YOLO 训练配置文件
│       ├── images/                      #     PNG 图像
│       └── labels/                      #     YOLO 格式 TXT 标注
├── outputs_task1/                       # 任务1 输出（可视化图表）
├── outputs_task2_yolov8s/               # YOLOv8s 训练输出
│   ├── predictions/                     #   测试集检测结果
│   └── runs_s/yolov8s_rfdd/            #   训练权重与指标
├── outputs_task2_yolov8m/               # YOLOv8m 训练输出
│   ├── predictions/                     #   测试集检测结果
│   └── runs_m/yolov8m_rfdd/            #   训练权重与指标
├── 日志yolov8s训练.txt                  # YOLOv8s 训练日志
├── 日志yolov8m训练.txt                  # YOLOv8m 训练日志
├── yolov8s.pt                           # YOLOv8s COCO 预训练权重
├── yolov8m.pt                           # YOLOv8m COCO 预训练权重
└── yolo26n.pt                           # YOLOv26n 预训练权重
```

---

## 数据集

**来源：** RFDD（Railway Fastener Defect Dataset）

**图像规格：** 2021 × 2048 像素，灰度图，存储为 HDF5 格式（训练验证集）和 PNG（测试集）

**标注格式：** YOLO 格式（`class_id center_x center_y width height`，归一化坐标）

**类别（6 类）：**

| 类别 ID | 英文名称    | 中文含义 |
|:------:|-----------|------|
|   0    | Deformed  | 变形  |
|   1    | Missing   | 缺失  |
|   2    | Displaced | 位移  |
|   3    | Inverted  | 反装  |
|   4    | Normal    | 正常  |
|   5    | Fractured | 断裂  |

**数据划分（去重后）：**

| 划分 | 图像数量 |
|:---:|:------:|
| 训练集 | ~1000 |
| 验证集 | ~250  |
| 测试集 | 100   |

**数据特点：**

- **严重类别不平衡：** Normal（正常）类别样本数远超所有缺陷类别之和（约 6:1）
- **小目标检测：** 扣件目标在高分辨率图像中占比很小（多数框面积小于 1%）
- **类间相似性：** Deformed（变形）、Displaced（位移）、Inverted（反装）视觉特征相似，容易混淆

---

## 环境配置

### 运行环境

- **操作系统：** Windows 11
- **GPU：** NVIDIA RTX 4060 Laptop（8GB VRAM）
- **Python：** 3.11.15（conda 环境 `pytorch`）
- **PyTorch：** 2.12.0 + CUDA 12.6
- **Ultralytics：** 8.4.52

### 依赖安装

```bash
# 创建 conda 环境
conda create -n pytorch python=3.11.15
conda activate pytorch

# 安装 PyTorch（CUDA 12.6）
（请在Pytorch官网获得适合PC的安装命令）

# 安装 ultralytics
conda install -c conda-forge ultralytics

# ultralytics官方文档
https://docs.ultralytics.com/zh

# 其他依赖
conda install h5py matplotlib numpy pillow
```

### 环境检查

```bash
python 检查python环境.py
```

---

## 实验流程

### 步骤 1：数据集预处理（去重）

```bash
python 1.数据集预处理.py
```

对原始 HDF5 数据集进行 MD5 哈希去重，移除完全重复的图像，并同步标注数据。
- 输入：`RFDD_datasets/train&val/`
- 输出：`Data_Prep/RFDD_clean/`

### 步骤 2：数据加载与可视化（任务 1）

```bash
python 2.图片读取.py
```

- 读取去重后数据，执行 80:20 分层抽样划分
- 可视化标注框（随机抽取样本）
- 统计类别分布并绘制柱状图
- 分析数据集问题（类别不平衡、小目标、类间相似性）
- 输出：`outputs_task1/`

### 步骤 3：YOLO 格式数据集构建

```bash
python 3.数据集分类.py
```

- 将 HDF5 数据导出为 PNG 图像（灰度 → RGB 三通道）
- 导出 YOLO 格式 TXT 标注文件
- 创建 `RFDD.yaml` 配置文件
- 输出：`Data_Prep/RFDD_Grouping/`

### 步骤 4：模型训练与评估（任务 2）

```bash
# 训练 YOLOv8s（推荐）
python 4.1yolo模型构建YOLOv8_small.py

# 训练 YOLOv8m
python 4.2yolo模型构建YOLOv8_m.py
```

训练超参数：

| 参数 | 值 |
|:---|:---|
| 输入尺寸 | 640×640 |
| Batch Size | 16 |
| Epoch | 100（Early Stop patience=20） |
| 学习率 | 0.001（cosine schedule, lrf=0.01） |
| 优化器 | AdamW |
| 预训练权重 | COCO |
| 数据增强 | Mosaic, HSV, Flip, Scale, Translate, Erasing |

### 步骤 5：模型推理

```bash
python 5.yolo模型调用.py
```

加载训练好的 `best.pt` 权重，在测试集上进行推理和评估。通过修改脚本中的 `MODEL_NAME` 变量切换 YOLOv8s / YOLOv8m。

---

## 模型对比

| 模型 | 参数量 | 预训练 | 推理速度 | 适用场景 |
|:---|:-----:|:-----:|:------:|:------:|
| YOLOv8s | ~11.2M | COCO | 快 | 轻量部署 |
| YOLOv8m | ~25.9M | COCO | 中等 | 精度优先 |

---

## 实验任务对照

| 任务 | 对应文件 | 内容 |
|:---:|------|------|
| 任务1 | `2.图片读取.py` | 数据准备与可视化 |
| 任务2 | `4.1yolo模型构建YOLOv8_small.py`、`4.2yolo模型构建YOLOv8_m.py` | YOLO 模型训练 |
| 任务3 | 待实现 | 非 YOLO 算法对比（Faster R-CNN / SSD 等） |
| 任务4 | 待实现 | 评价指标与结果分析报告 |

---

## 关键发现

1. **类别不平衡影响显著：** Normal 类占绝对主导，导致模型对缺陷类的关注度不足，需通过重采样或损失加权缓解
2. **小目标检测挑战：** 在 2021×2048 分辨率下，扣件目标尺寸较小，输入被缩放到 640×640 时信息丢失明显
3. **相似类别区分难度大：** Deformed、Displaced、Inverted 三类需要高分辨率的精细特征才能有效区分
4. **铁路安全场景的特殊性：** 缺陷检测中 Recall 比 Precision 更为关键，漏检（假阴性）可能导致安全事故
