"""
DualSG Meta Numerical Loader 
=====================================

仅用于 FIT 数据集的 **数值流（Meta）训练**。

返回：
(seq_x, seq_y, seq_x_mark, seq_y_mark,
 city_id, gender_id, age_id, element_id)
"""

import os
import json
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.logger import log


# ======================================================
# FIT 元数据映射表
# ======================================================
FIT_CITY_MAP = {
    'Cape Town': 0, 'Hong Kong': 1, 'London': 2, 'Los Angeles': 3,
    'Milan': 4, 'Moscow': 5, 'New York': 6, 'Paris': 7,
    'Rio de Janeiro': 8, 'Seoul': 9, 'Shanghai': 10, 'Singapore': 11,
    'Sydney': 12, 'Tokyo': 13
}

FIT_GENDER_MAP = {'female': 0, 'male': 1}

FIT_AGE_MAP = {
    'age_lt_18': 0,
    'age_18_25': 1,
    'age_25_40': 2,
    'age_gt_40': 3
}


class Dataset_DualSG_Fit_Num_Meta(Dataset):
    """
    FIT Meta Numerical Dataset
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
        seasonal_patterns=None,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2
    ):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.scale = scale
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

        # element 映射
        self.element_map = {}

        # 固定 meta size（FIT）
        self.num_cities = len(FIT_CITY_MAP)
        self.num_genders = len(FIT_GENDER_MAP)
        self.num_ages = len(FIT_AGE_MAP)

        self.__read_data__()

    # ==================================================
    # 读取并处理数据
    # ==================================================
    def __read_data__(self):
        self.scaler = StandardScaler()

        # ---------- 读取数据 ----------
        full_path = os.path.join(self.root_path, self.data_path)
        with open(full_path, "r") as f:
            full_data = json.load(f)

        # ---------- all 文件模式下切分 ----------
        data_list = full_data
        if "all" in self.data_path or not any(k in self.data_path for k in ["train", "val", "test"]):
            total = len(full_data)
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)

            if self.flag == "train":
                data_list = full_data[:train_size]
            elif self.flag == "val":
                data_list = full_data[train_size:train_size + val_size]
            else:
                data_list = full_data[train_size + val_size:]

        # ---------- 构建 element_map（全局一致） ----------
        for item in full_data:
            element = item.get("metadata", {}).get("element", "unknown")
            if element not in self.element_map:
                self.element_map[element] = len(self.element_map)

        # ---------- 提取数据 ----------
        self.series = []
        self.targets = []
        self.city_ids = []
        self.gender_ids = []
        self.age_ids = []
        self.element_ids = []

        for item in data_list:
            self.series.append(item["series"])
            self.targets.append(item["target"])

            meta = item.get("metadata", {})
            group = meta.get("group", "")
            element = meta.get("element", "unknown")

            # ===== 解析 FIT group =====
            city = "London"
            gender = "female"
            age = "age_25_40"

            for p in group.split("__"):
                if p.startswith("city:"):
                    city = p.split(":", 1)[1]
                elif p.startswith("gender:"):
                    gender = p.split(":", 1)[1]
                elif p.startswith("age:"):
                    age = p.split(":", 1)[1]

            self.city_ids.append(FIT_CITY_MAP.get(city, 0))
            self.gender_ids.append(FIT_GENDER_MAP.get(gender, 0))
            self.age_ids.append(FIT_AGE_MAP.get(age, 0))
            self.element_ids.append(self.element_map.get(element, 0))

        # ---------- numpy ----------
        self.series = np.array(self.series, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.city_ids = np.array(self.city_ids, dtype=np.int64)
        self.gender_ids = np.array(self.gender_ids, dtype=np.int64)
        self.age_ids = np.array(self.age_ids, dtype=np.int64)
        self.element_ids = np.array(self.element_ids, dtype=np.int64)

        # ---------- 标准化（FIT 使用 StandardScaler） ----------
        if self.scale:
            self.scaler.fit(self.series.reshape(-1, 1))
            self.series = self.scaler.transform(
                self.series.reshape(-1, 1)
            ).reshape(self.series.shape)
            self.targets = self.scaler.transform(
                self.targets.reshape(-1, 1)
            ).reshape(self.targets.shape)

        # ---------- 时间特征（占位） ----------
        self.data_stamp_x = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        self.data_stamp_y = np.arange(self.label_len + self.pred_len).reshape(-1, 1) / self.pred_len

        log(f"[FIT-Meta][{self.flag}] loaded {len(self.series)} samples")
        log(
            f"Elements:{len(self.element_map)}, "
            f"Cities:{self.num_cities}, "
            f"Genders:{self.num_genders}, "
            f"Ages:{self.num_ages}"
        )

    # ==================================================
    # Dataset API
    # ==================================================
    def __getitem__(self, index):
        seq_x = self.series[index].reshape(-1, 1)
        seq_y = self.targets[index].reshape(-1, 1)

        seq_x_mark = self.data_stamp_x
        seq_y_mark = self.data_stamp_y

        return (
            seq_x, seq_y,
            seq_x_mark, seq_y_mark,
            self.city_ids[index],
            self.gender_ids[index],
            self.age_ids[index],
            self.element_ids[index],
        )

    def __len__(self):
        return len(self.series)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(
            data.reshape(-1, 1)
        ).reshape(data.shape)
