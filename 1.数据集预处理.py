"""
RFDD 图像数据集降重程序
读取 RFDD_Train&val_Images.h5，检测并剔除重复图像。
支持精确去重（MD5哈希），使用逐张读取以控制内存占用。

用法：
    python 图片处理.py
"""

import h5py
import hashlib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import sys

# ---------- 路径配置 ----------
BASE_DIR = Path(__file__).parent   #自动获取当前脚本所在目录
H5_PATH = BASE_DIR /"RFDD_datasets"/"train&val" / "images" / "RFDD_Train&val_Images.h5" #原始.h5数据路径
OUTPUT_PATH = BASE_DIR /"Data_Prep"/"RFDD_clean"/ "train&val" / "images" / "RFDD_Train&val_Images_dedup.h5"  #去重后.h5图片数据路径
LABELS_PATH = BASE_DIR /"RFDD_datasets"/ "train&val" / "labels" / "RFDD_Train&val_Labels.h5"  #原始.h5标签数据路径
LABELS_OUTPUT_PATH = BASE_DIR /"Data_Prep"/"RFDD_clean"/ "train&val" / "labels" / "RFDD_Train&val_Labels_dedup.h5"  #去重后.h5标签数据路径


def compute_md5(img: np.ndarray) -> str:
    """对单张图像（numpy数组）计算 MD5 哈希"""
    return hashlib.md5(img.tobytes()).hexdigest()


def find_duplicates(h5_path: str):
    """
    逐张读取 H5 中的图像，计算 MD5 哈希并找出重复图像。
    不将全部图像加载到内存中。

    返回:
        hash_to_indices: {md5_hash: [索引列表]}
    """
    hash_to_indices = defaultdict(list)

    with h5py.File(h5_path, 'r') as f:
        images_ds = f['images']       # shape: (N, H, W)
        n = images_ds.shape[0]

        print(f"正在对 {n} 张图像（尺寸 {images_ds.shape[1]}×{images_ds.shape[2]}）"
              f" 进行 MD5 哈希计算...")
        for i in tqdm(range(n)):
            img = images_ds[i]        # 只读取当前这一张到内存
            h = compute_md5(img)
            hash_to_indices[h].append(i)

    return hash_to_indices


def write_dedup_dataset(h5_path: str, output_path: str,
                        keep_indices: list):
    """
    从原始 H5 文件中按保留的索引读取图像和文件名，写入新的 H5 文件。
    支持分块读写以控制内存占用。
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as src:
        images_ds = src['images']
        filenames_ds = src['filenames']
        H, W = images_ds.shape[1], images_ds.shape[2]
        keep = np.array(sorted(keep_indices), dtype=np.int64)
        m = len(keep)

        print(f"\n正在将 {m} 张去重后的图像写入 {output_path} ...")

        with h5py.File(output_path, 'w') as dst:
            dst_images = dst.create_dataset(
                'images', shape=(m, H, W), dtype=images_ds.dtype, chunks=True
            )
            dst_filenames = dst.create_dataset(
                'filenames', shape=(m,), dtype=filenames_ds.dtype
            )

            # 分批写入，避免瞬时大内存
            chunk_size = 100
            for start in tqdm(range(0, m, chunk_size)):
                end = min(start + chunk_size, m)
                idx_block = keep[start:end]
                dst_images[start:end] = images_ds[idx_block.tolist()]
                dst_filenames[start:end] = filenames_ds[idx_block.tolist()]

    print(f"去重图像保存完成 → {output_path}")


def sync_labels(unique_filenames: list, labels_path: str, output_path: str):
    """
    根据去重后保留的图像文件名，同步对应的标签数据。
    """
    if not Path(labels_path).exists():
        print(f"\n标签文件 {labels_path} 不存在，跳过标签同步。")
        return

    print(f"\n正在同步标签数据...")
    with h5py.File(labels_path, 'r') as f:
        label_filenames = f['filenames'][:]
        labels = f['labels'][:]

    # 统一转为字符串
    def to_str(fn):
        return fn.decode('utf-8', errors='replace') if isinstance(fn, bytes) else str(fn)

    unique_set = {to_str(fn) for fn in unique_filenames}

    matched_indices = []
    for i, fn in enumerate(label_filenames):
        if to_str(fn) in unique_set:
            matched_indices.append(i)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('labels', data=labels[matched_indices], chunks=True)
        f.create_dataset('filenames', data=label_filenames[matched_indices])

    print(f"标签同步完成：{len(matched_indices)} 条记录 → {output_path}")
    if len(matched_indices) < len(label_filenames):
        print(f"  注意：{len(label_filenames) - len(matched_indices)} 条标签记录因对应图像被去重而移除。")


def main():
    if not H5_PATH.exists():
        print(f"错误：找不到文件 {H5_PATH}")
        sys.exit(1)

    # ----- 1. 扫描去重 -----
    print("=" * 55)
    print("  RFDD 图像去重")
    print("  方法: MD5 精确哈希")
    print("=" * 55)

    hash_to_indices = find_duplicates(H5_PATH)

    n_total = sum(len(v) for v in hash_to_indices.values())
    n_unique = len(hash_to_indices)
    dup_groups = {h: idxs for h, idxs in hash_to_indices.items() if len(idxs) > 1}
    n_dup = n_total - n_unique
    keep_indices = [idxs[0] for idxs in hash_to_indices.values()]

    # ----- 2. 打印报告 -----
    print(f"\n{'='*55}")
    print(f"  去重分析报告")
    print(f"{'='*55}")
    print(f"  原始图像数量:   {n_total}")
    print(f"  唯一图像数量:   {n_unique}")
    print(f"  重复图像数量:   {n_dup}  ({n_dup/n_total*100:.2f}%)" if n_total else "")
    print(f"  重复组数:       {len(dup_groups)}")
    if dup_groups:
        print(f"  最大重复组:     {max(len(v) for v in dup_groups.values())} 张")
        print(f"\n  重复详情（前 10 组）:")
        # 获取文件名用于显示
        with h5py.File(H5_PATH, 'r') as f:
            fn_ds = f['filenames']
            for h, idxs in list(dup_groups.items())[:10]:
                names = [str(fn_ds[i]) for i in idxs]
                print(f"    [{h[:12]}...] {len(idxs)} 个副本: {names}")
    print(f"{'='*55}")

    # ----- 3. 写入去重结果 -----
    write_dedup_dataset(H5_PATH, OUTPUT_PATH, keep_indices)

    # ----- 4. 同步标签 -----
    with h5py.File(H5_PATH, 'r') as f:
        unique_fns = [f['filenames'][i] for i in keep_indices]
    sync_labels(unique_fns, LABELS_PATH, LABELS_OUTPUT_PATH)

    print(f"\n{'='*55}")
    print(f"  去重任务完成！")
    print(f"  去重前: {n_total} 张 → 去重后: {n_unique} 张")
    print(f"  图像输出: {OUTPUT_PATH}")
    if LABELS_PATH.exists():
        print(f"  标签输出: {LABELS_OUTPUT_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
