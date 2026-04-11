"""
FIT融合数据集加载器

该模块实现了DualSG_Fit_Fusion模型的数据集加载器，支持FIT数据集的处理和加载。

主要功能：
1. 支持FIT数据集的加载和处理
2. 支持自动70/10/20数据切分
3. 支持元数据（city / gender / age / element）的处理
4. 支持caption embedding的加载和对齐
5. 支持数据标准化

返回格式：
  seq_x:[seq_len,1], seq_y:[pred_len,1],
  seq_x_mark:[seq_len,1], seq_y_mark:[pred_len,1],
  city_id, gender_id, age_id, element_id,
  caption_emb:[llm_dim]

依赖：
- os
- json
- numpy
- torch
- sklearn.preprocessing.StandardScaler
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.logger import log


# ======================================================
# FIT 元数据映射表
# ======================================================
CITY_MAP = {
    'Cape Town': 0, 'Hong Kong': 1, 'London': 2, 'Los Angeles': 3,
    'Milan': 4, 'Moscow': 5, 'New York': 6, 'Paris': 7,
    'Rio de Janeiro': 8, 'Seoul': 9, 'Shanghai': 10, 'Singapore': 11,
    'Sydney': 12, 'Tokyo': 13
}

GENDER_MAP = {'female': 0, 'male': 1}

AGE_MAP = {
    'age_lt_18': 0,
    'age_18_25': 1,
    'age_25_40': 2,
    'age_gt_40': 3
}


# ======================================================
# 工具函数
# ======================================================
from utils.data_provider_utils import is_all_mode, parse_fit_group, generate_time_features


# ======================================================
# Dataset
# ======================================================
class Dataset_DualSG_Fit_Fusion(Dataset):
    """
    FIT Fusion 数据集类

    用于加载和处理FIT数据集，支持元数据和caption embedding。

    参数:
        root_path: 数据根目录
        flag: 数据类型，可选值：train / val / test
        size: 序列长度配置，格式为(seq_len, label_len, pred_len)
        features: 特征类型，默认为"S"（单变量）
        data_path: 数据文件路径
        target: 目标变量，默认为"trend"
        scale: 是否进行数据标准化
        timeenc: 时间编码方式
        freq: 时间频率
        caption_emb_path: caption embedding文件路径
        seasonal_patterns: 季节模式
        split_ratio: 数据切分比例，默认为(0.7, 0.1, 0.2)
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="fit_dualsg_all.json",
        target="trend",
        scale=True,
        timeenc=0,
        freq="w",
        caption_emb_path=None,
        seasonal_patterns=None,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    ):
        """
        初始化FIT Fusion数据集

        参数:
            root_path: 数据根目录
            flag: 数据类型，可选值：train / val / test
            size: 序列长度配置，格式为(seq_len, label_len, pred_len)
            features: 特征类型，默认为"S"（单变量）
            data_path: 数据文件路径
            target: 目标变量，默认为"trend"
            scale: 是否进行数据标准化
            timeenc: 时间编码方式
            freq: 时间频率
            caption_emb_path: caption embedding文件路径
            seasonal_patterns: 季节模式
            split_ratio: 数据切分比例，默认为(0.7, 0.1, 0.2)

        异常:
            AssertionError: 如果flag不在['train', 'val', 'test']中
            AssertionError: 如果caption_emb_path为None
        """
        assert flag in ["train", "val", "test"]
        assert caption_emb_path is not None, "caption_emb_path must be provided!"

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.caption_emb_path = caption_emb_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 序列长度
        if size is None:
            # 使用配置的默认值
            self.seq_len = 48
            self.label_len = 0
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # 固定 FIT meta size
        self.num_cities = len(CITY_MAP)
        self.num_genders = len(GENDER_MAP)
        self.num_ages = len(AGE_MAP)

        self.scaler = StandardScaler()
        self.element_map = {}

        # split 控制
        self.is_all_mode = is_all_mode(self.data_path)
        self.total = None
        self.train_size = None
        self.val_size = None

        self.__read_data__()
        self.__load_caption_embedding__()

    # ==================================================
    # 读取并处理数据
    # ==================================================
    def __read_data__(self):
        """
        读取并处理数据

        步骤：
        1. 读取JSON数据
        2. 根据模式进行数据切分
        3. 构建元素映射表
        4. 提取数据和元数据
        5. 转换为numpy数组
        6. 数据标准化
        7. 生成时间特征
        """
        # 1) 读取 JSON
        full_path = os.path.join(self.root_path, self.data_path)
        with open(full_path, "r") as f:
            full_data = json.load(f)

        # 2) split（仅对 all 模式）
        self.total = len(full_data)
        self.train_size = int(self.total * self.train_ratio)
        self.val_size = int(self.total * self.val_ratio)

        if self.is_all_mode:
            if self.flag == "train":
                data_list = full_data[:self.train_size]
            elif self.flag == "val":
                data_list = full_data[self.train_size:self.train_size + self.val_size]
            else:
                data_list = full_data[self.train_size + self.val_size:]
        else:
            data_list = full_data

        # 3) 构建 element_map（基于 full_data，保证一致）
        for item in full_data:
            element = item.get("metadata", {}).get("element", "unknown")
            if element not in self.element_map:
                self.element_map[element] = len(self.element_map)

        # 4) 提取数据
        self.series, self.targets = [], []
        self.city_ids, self.gender_ids, self.age_ids, self.element_ids = [], [], [], []

        for item in data_list:
            self.series.append(item["series"])
            self.targets.append(item["target"])

            meta = item.get("metadata", {})
            group = meta.get("group", "")
            element = meta.get("element", "unknown")

            city, gender, age = _parse_fit_group(group)

            self.city_ids.append(CITY_MAP.get(city, 0))
            self.gender_ids.append(GENDER_MAP.get(gender, 0))
            self.age_ids.append(AGE_MAP.get(age, 0))
            self.element_ids.append(self.element_map.get(element, 0))

        # 5) numpy
        self.series = np.array(self.series, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.city_ids = np.array(self.city_ids, dtype=np.int64)
        self.gender_ids = np.array(self.gender_ids, dtype=np.int64)
        self.age_ids = np.array(self.age_ids, dtype=np.int64)
        self.element_ids = np.array(self.element_ids, dtype=np.int64)

        # 6) 标准化（FIT 使用 StandardScaler）
        if self.scale:
            self.scaler.fit(self.series.reshape(-1, 1))
            self.series = self.scaler.transform(
                self.series.reshape(-1, 1)
            ).reshape(self.series.shape)
            self.targets = self.scaler.transform(
                self.targets.reshape(-1, 1)
            ).reshape(self.targets.shape)

        # 7) 时间特征（占位）
        self.data_stamp_x, self.data_stamp_y = generate_time_features(self.seq_len, self.label_len, self.pred_len)

        log(f"[FIT-Fusion][{self.flag}] loaded {len(self.series)} samples")
        log(
            f"  Elements:{len(self.element_map)}, "
            f"Cities:{self.num_cities}, "
            f"Genders:{self.num_genders}, "
            f"Ages:{self.num_ages}"
        )

    # ==================================================
    # caption embedding 对齐
    # ==================================================
    def __load_caption_embedding__(self):
        """
        加载并对齐caption embedding

        步骤：
        1. 加载caption embedding文件
        2. 确保数据类型为torch.Tensor
        3. 根据模式进行数据切分和对齐
        4. 转换为float类型

        异常:
            ValueError: 如果caption_emb数量与数据数量不匹配
        """
        log(f"[Fusion] Loading caption embedding from: {self.caption_emb_path}")
        full_emb = torch.load(self.caption_emb_path)

        if not isinstance(full_emb, torch.Tensor):
            full_emb = torch.tensor(full_emb)

        if self.is_all_mode:
            if full_emb.shape[0] != self.total:
                raise ValueError(
                    f"caption_emb count {full_emb.shape[0]} != json count {self.total}"
                )

            if self.flag == "train":
                emb = full_emb[:self.train_size]
            elif self.flag == "val":
                emb = full_emb[self.train_size:self.train_size + self.val_size]
            else:
                emb = full_emb[self.train_size + self.val_size:]
        else:
            if full_emb.shape[0] != len(self.series):
                raise ValueError(
                    f"caption_emb size {full_emb.shape[0]} != series size {len(self.series)}"
                )
            emb = full_emb

        self.caption_emb = emb.float()
        log(f"[Fusion] caption_emb aligned: {self.caption_emb.shape}")

    # ==================================================
    # Dataset API
    # ==================================================
    def __getitem__(self, index):
        """
        获取指定索引的数据

        参数:
            index: 数据索引

        返回:
            tuple: 包含以下元素的元组：
                - series: 输入序列，形状为[seq_len, 1]
                - targets: 目标序列，形状为[pred_len, 1]
                - data_stamp_x: 输入时间戳，形状为[seq_len, 1]
                - data_stamp_y: 目标时间戳，形状为[pred_len, 1]
                - city_id: 城市ID
                - gender_id: 性别ID
                - age_id: 年龄ID
                - element_id: 元素ID
                - caption_emb: 文本嵌入，形状为[llm_dim]
        """
        return (
            self.series[index].reshape(-1, 1),
            self.targets[index].reshape(-1, 1),
            self.data_stamp_x,
            self.data_stamp_y,
            self.city_ids[index],
            self.gender_ids[index],
            self.age_ids[index],
            self.element_ids[index],
            self.caption_emb[index],
        )

    def __len__(self):
        """
        获取数据集长度

        返回:
            int: 数据集样本数量
        """
        return len(self.series)

    def inverse_transform(self, data):
        """
        数据反标准化

        参数:
            data: 标准化后的数据

        返回:
            numpy.ndarray: 反标准化后的数据
        """
        return self.scaler.inverse_transform(
            data.reshape(-1, 1)
        ).reshape(data.shape)
