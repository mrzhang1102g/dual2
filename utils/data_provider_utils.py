"""数据加载辅助工具。"""

import os

import numpy as np


def is_all_mode(data_path: str) -> bool:
    """判断当前文件是否需要在内部做 70/10/20 切分。"""
    base = os.path.basename(data_path)
    if "all" in base:
        return True
    if not any(token in base for token in ["train", "val", "test"]):
        return True
    return False


def split_data(total: int, flag: str, train_ratio: float, val_ratio: float):
    """按顺序切分 train / val / test 的索引范围。"""
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    if flag == "train":
        start, end = 0, train_size
    elif flag == "val":
        start, end = train_size, train_size + val_size
    else:
        start, end = train_size + val_size, total

    return start, end, train_size, val_size


def build_metadata_maps(full_data):
    """从完整数据中构建 element/group 的全局映射。"""
    element_map = {}
    group_map = {}

    for item in full_data:
        meta = item.get("metadata", {})
        element = meta.get("element", "unknown")
        group = meta.get("group", "unknown")

        if element not in element_map:
            element_map[element] = len(element_map)
        if group not in group_map:
            group_map[group] = len(group_map)

    return element_map, group_map


def generate_time_features(seq_len, label_len, pred_len):
    """生成简单的位置型时间特征。"""
    data_stamp_x = np.arange(seq_len).reshape(-1, 1) / seq_len
    data_stamp_y = np.arange(label_len + pred_len).reshape(-1, 1) / pred_len
    return data_stamp_x, data_stamp_y


def parse_fit_group(group: str):
    """解析 FIT 的 group 字符串，拆出 city / gender / age。"""
    city, gender, age = "London", "female", "age_25_40"  # 缺省值只用于兜底。
    for part in group.split("__"):
        if part.startswith("city:"):
            city = part.split(":", 1)[1]
        elif part.startswith("gender:"):
            gender = part.split(":", 1)[1]
        elif part.startswith("age:"):
            age = part.split(":", 1)[1]
    return city, gender, age
