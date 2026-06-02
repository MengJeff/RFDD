# RFDD — 铁路扣件缺陷检测 (Railway Fastener Defect Detection)

基于 YOLOv8 的铁路扣件状态检测系统，使用 RFDD（Railway Fastener Defect Dataset）数据集训练和评估目标检测模型，识别扣件的六种状态：变形、缺失、位移、反装、正常、断裂。

# 注意事项

1. 将仓库克隆到本地后请将解压后的数据集整体复制到本地仓库根目录下，依次按下方指引运行程序即可，没有的文件夹会自动生成或下载（推荐挂代理）；YAML配置文件在训练前置步骤完成后会自动生成，见"...\Data_Prep\RFDD_Grouping\RFDD".yaml"；请在硬盘中至少留出20GB空间。
2. 如果已有已训练的模型权重文件，请将文件复制进对应结果文件夹（文件夹名称详见项目结构）。
3. .cache文件为训练模型时yolo自动生成的缓存文件，相当于索引使得文件读取更加快速。每次训练模型时，最好将此缓存文件删掉。
4. 模型训练时，该ultralytics库会自动求解最优参数，使得先前输入的超参数被覆盖。对应参数“optimizer (str): 选择优化器。可选值包括 'SGD', 'Adam', 'AdamW', 'RMSProp' 等，或 'auto'（由系统自动选择为AdamW）”。系统最终选择的优化器为 AdamW，初始学习率为 0.001，动量为 0.9。同时，它将模型参数分为 3 个组（无衰减权重、有衰减权重、偏置值）分别进行梯度更新。
5. 在自定义 YOLO 的输入分辨率时，必须遵守一个基本原则：输入图像的宽度和高度必须是 32 的整数倍目前640*640属于中等分辨率，可能会丢失信息，增大该超参数，显存会增加，请谨慎考虑该参数与batch参数的配合，避免爆显存。
6. 针对目前问题“Normal 类样本远多于缺陷类样本”，会导致以下问题：由于“Normal”样本数量庞大，其产生的损失值（Loss）会主导梯度的更新方向。模型倾向于简单地将所有区域预测为“Normal”以快速降低整体损失，从而抑制了对稀有缺陷特征的学习。解决方法很多，目前在考虑“开启或调整Focal Loss（在超参数配置文件中设置 fl_gamma，例如设为 fl_gamma=1.5 或 2.0，Focal Loss通过降低易分类样本（如大量的Normal）的权重，使模型专注于难分类的样本。

---

# 项目结构

```
RFDD/
├── README.md                                               #本文件
├── .gitignore                                              #仓库忽略文件
├── 0.检查python环境.py                                     #环境检查：PyTorch/CUDA/ultralytics 版本
├── 1.数据集预处理.py                                       #数据集去重（MD5 哈希）
├── 2.图片读取.py                                           #任务1：数据加载、标注可视化、统计分析
├── 3.数据集分类.py                                         #数据集划分与 YOLO 格式导出
├── 4.1yolo模型构建YOLOv8_small.py                          #任务2：YOLOv8s 训练与评估
├── 4.2yolo模型构建YOLOv8_m.py                              #任务2：YOLOv8m 训练与评估
├── 5.1FasterR-CNN模型构建fasterrcnn_resnet50_fpn_v2.py     #任务3：Faster R-CNN 训练与评估
├── 5.2RetinaNet模型构建retinanet_resnet50_fpn_v2.py        #任务3：RetinaNet 训练与评估
├── RFDD_datasets/                                          #原始数据集（含占位符）
│   ├── train&val/                                              #- 训练+验证集（复制解压后文件即可）
│   └── test/                                                   #- 测试集（复制解压后文件即可）
├── Data_Prep/                                              #预处理后数据（含占位符）
│   ├── RFDD_clean/                                             #- 去重后的 H5 数据
│   └── RFDD_Grouping/                                          #- YOLO 格式数据集（train/val/test）
│       ├── RFDD.yaml                                               #YOLO 训练配置文件
│       └── ...                                                     #其他略
├── outputs_task1/                                          #任务1输出（运行脚本后自动生成内容）
│   ├── vis_boxes_Train.png                                     #训练集标注框可视化（5 张）
│   ├── vis_boxes_Val.png                                       #验证集标注框可视化（5 张）
│   ├── vis_boxes_Test.png                                      #测试集标注框可视化（5 张）
│   ├── class_distribution.png                                  #全类别分布柱状图
│   ├── class_distribution_defects_only.png                     #缺陷类分布柱状图（剔除 Normal）
│   └── bbox_area_histogram.png                                 #边界框面积分布直方图
├── outputs_task2_yolov8s/                                  #任务2YOLOv8s训练输出（运行脚本后自动生成内容）
│   ├── predictions/                                            #测试集检测结果
│   ├── runs_s/yolov8s_rfdd/                                    #训练后生成的权重与指标
│   ├── log_yolov8s_yyyymmdd_hhmmss.txt                         #训练日志
│   └── training_curves.png                                     #训练曲线
├── outputs_task2_yolov8m/                                  #任务2YOLOv8m训练输出（运行脚本后自动生成内容）
│   └── ...                                                     #结构类似，略
├── outputs_task3_fasterrcnn_resnet50_fpn_v2/               #任务3Faster R-CNN训练输出（运行脚本后自动生成内容）
│   └── ...                                                     #结构类似，略
├── outputs_task3_retinanet_resnet50_fpn_v2/                #任务3RetinaNet训练输出（运行脚本后自动生成内容）
│   └── ...                                                     #结构类似，略
├── yolov8s.pt                                              #YOLOv8s COCO 预训练权重（运行脚本后自动下载）
├── yolov8m.pt                                              #YOLOv8m COCO 预训练权重（运行脚本后自动下载）
└── yolo26n.pt                                              #YOLOv26n 预训练权重（运行脚本后自动下载）
```

---

# 数据集概述

**来源：** RFDD（Railway Fastener Defect Dataset）

**图像规格：** 2021 × 2048 像素，灰度图，存储为 HDF5 格式（训练验证集）和 PNG（测试集）

**标注格式：** YOLO 格式（`class_id center_x center_y width height`，归一化坐标）

**类别（6 类）：**

| 类别 ID | 英文名称  | 中文含义 |
| :-----: | --------- | -------- |
|    0    | Deformed  | 变形     |
|    1    | Fractured | 断裂     |
|    2    | Missing   | 缺失     |
|    3    | Inverted  | 反装     |
|    4    | Normal    | 正常     |
|    5    | Displaced | 位移     |

**数据划分（去重后）：**

|  划分  | 图像数量 |
| :----: | :------: |
| 训练集 |  ~1000  |
| 验证集 |   ~250   |
| 测试集 |   100   |

**数据特点：**

- **严重类别不平衡：** Normal（正常）类别样本数远超所有缺陷类别之和（约 6:1）
- **小目标检测：** 扣件目标在高分辨率图像中占比很小（多数框面积小于 1%）
- **类间相似性：** Deformed（变形）、Displaced（位移）、Inverted（反装）视觉特征相似，容易混淆

---

# 环境配置

## 运行环境

- **操作系统：** Windows 11
- **GPU：** NVIDIA RTX 4060 Laptop（8GB VRAM）及以上
- **RAM：** 16GB及以上
- **Python：** 3.10.20及以上（conda 环境）
- **PyTorch：** 2.7.0 + CUDA 12.8及以上（视具体情况而定即可）
- **Ultralytics：** 8.4.52及以上

## 依赖安装

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

---

# 实验流程

## 步骤 0：检查python环境```bash

python 1.数据集预处理.py

```bash
python 0.检查python环境.py
```

根据控制台输出的提示完成相应操作即可。

## 步骤 1：数据集预处理（去重）

```bash
python 1.数据集预处理.py
```

对原始 HDF5 数据集进行 MD5 哈希去重，移除完全重复的图像，并同步标注数据。

- 输入：`RFDD_datasets/train&val/`
- 输出：`Data_Prep/RFDD_clean/`

## 步骤 2：数据加载与可视化（任务 1）

```bash
python 2.图片读取.py
```

- 读取去重后数据，执行 80:20 分层抽样划分
- 可视化标注框（随机抽取样本）
- 统计类别分布并绘制柱状图
- 分析数据集问题（类别不平衡、小目标、类间相似性）
- 输出：`outputs_task1/`

## 步骤 3：YOLO 格式数据集构建

```bash
python 3.数据集分类.py
```

- 将 HDF5 数据导出为 PNG 图像（灰度 → RGB 三通道）
- 导出 YOLO 格式 TXT 标注文件
- 创建 `RFDD.yaml` 配置文件
- 输出：`Data_Prep/RFDD_Grouping/`

## 步骤 4：模型训练与评估（任务 2）

```bash
# 训练 YOLOv8s
python 4.1yolo模型构建YOLOv8_small.py

# 训练 YOLOv8m
python 4.2yolo模型构建YOLOv8_m.py
```

## 步骤 5：模型对比（任务 3）

```bash
# 训练 Faster R-CNN
python 5.1FasterR-CNN模型构建fasterrcnn_resnet50_fpn_v2.py

# 训练 RetinaNet
python 5.2RetinaNet模型构建retinanet_resnet50_fpn_v2.py
```

训练超参数参见实验报告[XiaoZhefu/RFDD-LaTeX-Report](https://github.com/XiaoZhefu/RFDD-LaTeX-Report)
