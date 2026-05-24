"""
RFDD 项目环境检查
=================
检查当前 Python 环境是否满足所有程序运行所需的条件：
  - Python 版本
  - 第三方依赖库
  - PyTorch / CUDA / GPU
  - ultralytics 框架
  - 项目数据文件完整性
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

print("=" * 55)
print("  RFDD 项目环境检查")
print("=" * 55)

# ===================== 1. Python 版本 =====================
print(f"\n{'─' * 45}")
print("  [1] Python 版本")
print(f"{'─' * 45}")
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"  Python: {py_ver}")
if sys.version_info < (3, 9):
    print("  [警告] 建议 Python >= 3.9")

# ===================== 2. 第三方库检查 =====================
print(f"\n{'─' * 45}")
print("  [2] 第三方依赖库")
print(f"{'─' * 45}")

required_libs = {
    "torch":       "PyTorch（深度学习框架）",
    "numpy":       "NumPy（数值计算）",
    "h5py":        "h5py（HDF5 数据读取）",
    "PIL":         "Pillow（图像处理）",
    "matplotlib":  "Matplotlib（绘图）",
    "yaml":        "PyYAML（YAML 配置解析）",
    "tqdm":        "tqdm（进度条）",
    "ultralytics": "Ultralytics（YOLO 训练框架）",
}

all_ok = True
for lib_name, desc in required_libs.items():
    try:
        mod = __import__(lib_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {lib_name:<20s} {str(ver):<12s}  # {desc}")
    except ImportError:
        print(f"  [缺失] {lib_name:<20s} {'未安装':<12s}  # {desc}")
        all_ok = False

if not all_ok:
    print("\n  请安装缺失的库:")
    print("  conda install torch numpy h5py pillow matplotlib pyyaml tqdm ultralytics")

# ===================== 3. PyTorch 与 CUDA =====================
print(f"\n{'─' * 45}")
print("  [3] PyTorch 与 CUDA 环境")
print(f"{'─' * 45}")

import torch

print(f"  PyTorch 版本:     {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"  CUDA 可用:        {cuda_available}")

if cuda_available:
    print(f"  CUDA 版本:        {torch.version.cuda}")
    print(f"  cuDNN 版本:       {torch.backends.cudnn.version()}")
    print(f"  GPU 数量:         {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        name = torch.cuda.get_device_name(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  GPU[{i}]:            {name}")
        print(f"    显存:            {mem_gb:.1f} GB")
        print(f"    计算能力:        {props.major}.{props.minor}")
    # 简单的 GPU 张量运算测试
    try:
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = a @ b
        print(f"  GPU 运算测试:     通过")
    except Exception as e:
        print(f"  GPU 运算测试:     失败 ({e})")
else:
    print("  [警告] 当前无可用 GPU，训练过程将非常缓慢")

# ===================== 4. ultralytics 框架 =====================
print(f"\n{'─' * 45}")
print("  [4] Ultralytics 框架")
print(f"{'─' * 45}")

try:
    import ultralytics
    from ultralytics import YOLO  # noqa: F401 — 验证 YOLO 类可正常导入
    print(f"  Ultralytics 版本: {ultralytics.__version__}")
    print(f"  YOLO 类导入:      成功")
except ImportError:
    print(f"  [缺失] ultralytics 未安装")
    print("  请运行: conda install ultralytics")
except Exception as e:
    print(f"  [错误] {e}")

# ===================== 5. 数据集文件检查 =====================
print(f"\n{'─' * 45}")
print("  [5] 数据集文件")
print(f"{'─' * 45}")

data_checks = {
    "训练集 H5 (images)":
        BASE_DIR / "RFDD_datasets" / "train&val" / "images" / "RFDD_Train&val_Images.h5",
    "训练集 H5 (labels)":
        BASE_DIR / "RFDD_datasets" / "train&val" / "labels" / "RFDD_Train&val_Labels.h5",
    "测试集图像目录":
        BASE_DIR / "RFDD_datasets" / "test" / "images",
    "测试集标签目录":
        BASE_DIR / "RFDD_datasets" / "test" / "labels",
}

for desc, path in data_checks.items():
    if path.exists():
        print(f"  [OK] {desc}")
        print(f"        {path}")
    else:
        print(f"  [缺失] {desc}")
        print(f"          {path}")

# ===================== 6. 预训练权重检查 =====================
print(f"\n{'─' * 45}")
print("  [6] 预训练权重文件")
print(f"{'─' * 45}")

weight_files = ["yolov8s.pt", "yolov8m.pt"]
for wf in weight_files:
    wf_path = BASE_DIR / wf
    if wf_path.exists():
        size_mb = wf_path.stat().st_size / (1024**2)
        print(f"  [OK] {wf}  ({size_mb:.1f} MB)")
    else:
        print(f"  [提示] {wf} 未找到，训练脚本首次运行时会自动下载")

# ===================== 7. 已训练权重检查 =====================
print(f"\n{'─' * 45}")
print("  [7] 已训练模型权重")
print(f"{'─' * 45}")

trained_weights = [
    ("YOLOv8s best.pt", BASE_DIR / "outputs_task2_yolov8s" / "runs_s" / "yolov8s_rfdd" / "weights" / "best.pt"),
    ("YOLOv8m best.pt", BASE_DIR / "outputs_task2_yolov8m" / "runs_m" / "yolov8m_rfdd" / "weights" / "best.pt"),
]

for label, path in trained_weights:
    if path.exists():
        print(f"  [OK] {label}")
    else:
        print(f"  [未训练] {label} — 请先运行训练脚本")

# ===================== 汇总 =====================
print(f"\n{'=' * 55}")
print("  检查完成")
print(f"{'=' * 55}")
print(f"  项目根目录: {BASE_DIR}")
print(f"  Python:     {py_ver}  |  PyTorch: {torch.__version__}  |  CUDA: {'可用' if cuda_available else '不可用'}")
