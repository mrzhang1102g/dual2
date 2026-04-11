import os
import json
import numpy as np


def is_all_mode(data_path: str) -> bool:
    """
    判断是否是全量文件模式（需要内部做70/10/20切分）

    参数:
        data_path: 数据文件路径

    返回:
        bool: 如果是全量文件模式返回True，否则返回False
    """
    base = os.path.basename(data_path)
    if "all" in base:
        return True
    if not any(k in base for k in ["train", "val", "test"]):
        return True
    return False


def split_data(total: int, flag: str, train_ratio: float, val_ratio: float):
    """
    数据切分规则：
    train: [0, train_ratio)
    val  : [train_ratio, train_ratio + val_ratio)
    test : [train_ratio + val_ratio, 1.0)

    参数:
        total: 总样本数
        flag: 数据类型，可选值：train / val / test
        train_ratio: 训练数据比例
        val_ratio: 验证数据比例

    返回:
        tuple: (start, end, train_size, val_size)
    """
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
    """
    构建元数据映射表

    参数:
        full_data: 完整的数据集

    返回:
        tuple: (element_map, group_map)
    """
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
    """
    生成时间特征

    参数:
        seq_len: 输入序列长度
        label_len: 标签序列长度
        pred_len: 预测序列长度

    返回:
        tuple: (data_stamp_x, data_stamp_y)
    """
    # x: [0..seq_len-1]/seq_len
    data_stamp_x = np.arange(seq_len).reshape(-1, 1) / seq_len
    # y: [0..(label_len+pred_len)-1]/pred_len
    data_stamp_y = np.arange(label_len + pred_len).reshape(-1, 1) / pred_len

    return data_stamp_x, data_stamp_y


def parse_fit_group(group: str):
    """
    解析 FIT group 字符串

    示例: "age:age_25_40__city:Hong Kong__gender:female"

    参数:
        group: 包含城市、性别和年龄信息的字符串

    返回:
        tuple: (city, gender, age) 解析后的城市、性别和年龄信息
    """
    city, gender, age = "London", "female", "age_25_40"  # 默认兜底
    for p in group.split("__"):
        if p.startswith("city:"):
            city = p.split(":", 1)[1]
        elif p.startswith("gender:"):
            gender = p.split(":", 1)[1]
        elif p.startswith("age:"):
            age = p.split(":", 1)[1]
    return city, gender, age
