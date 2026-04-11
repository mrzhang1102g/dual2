"""
DualSG GeoStyle Numerical Loader
==============================================

用于 GeoStyle 数据集的 数值流训练（DualSG）。

GeoStyle 样本格式:
{
    "series": [...52],
    "target": [...26],
    "annotations": "...",
    "metadata": {
        "element": "clothing_pattern__Graphics",
        "group": "06",                 # City / Group
        "norm": [...],                 # [min_v, max_v, eps]
    }
}

返回：
(seq_x, seq_y, seq_x_mark, seq_y_mark, element_id, group_id, norm)
"""

import os
import json
import numpy as np
from torch.utils.data import Dataset
from utils.logger import log


class Dataset_DualSG_Geo_Num_Meta(Dataset):
    """
    GeoStyle Numerical Dataset
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="geo_dualsg_all.json",
        target="trend",
        scale=True,
        timeenc=0,
        freq="w",
        seasonal_patterns=None,
        use_element=False,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    ):
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.use_element = use_element
        self.scale = scale
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # GeoStyle 序列长度
        if size is None:
            # 使用配置的默认值
            self.seq_len = 52
            self.label_len = 0
            self.pred_len = 26
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # ===== 元数据映射表 =====
        self.element_map = {}
        self.group_map = {}

        self.__read_data__()

    # ==================================================
    # 读取并处理数据
    # ==================================================
    def __read_data__(self):

        # ---------- 读取 json ----------
        full_path = os.path.join(self.root_path, self.data_path)
        with open(full_path, "r") as f:
            full_data = json.load(f)

        # ---------- 数据切分（all.json 模式） ----------
        data_list = full_data
        if "all" in self.data_path or not any(
            k in self.data_path for k in ["train", "val", "test"]
        ):
            total = len(full_data)
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)

            if self.flag == "train":
                data_list = full_data[:train_size]
            elif self.flag == "val":
                data_list = full_data[train_size:train_size + val_size]
            else:
                data_list = full_data[train_size + val_size:]

        # ---------- 构建 element / group 全局映射 ----------
        for item in full_data:
            meta = item.get("metadata", {})

            element = meta.get("element", "unknown")
            group = meta.get("group", "unknown")

            if element not in self.element_map:
                self.element_map[element] = len(self.element_map)
            if group not in self.group_map:
                self.group_map[group] = len(self.group_map)

        # ---------- 提取样本 ----------
        self.series = []
        self.targets = []
        self.element_ids = []
        self.group_ids = []
        self.norms = []

        for item in data_list:
            meta = item.get("metadata", {})

            # 数值序列
            self.series.append(item["series"])
            self.targets.append(item["target"])

            # element / group → id
            element = meta.get("element", "unknown")
            group = meta.get("group", "unknown")

            self.element_ids.append(self.element_map.get(element, 0))
            self.group_ids.append(self.group_map.get(group, 0))

            # per-series 归一化参数（用于 test 阶段反归一化）
            norm = meta.get("norm", [0.0, 1.0, 0.0])
            self.norms.append(norm)

        # ---------- numpy ----------
        self.series = np.array(self.series, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.element_ids = np.array(self.element_ids, dtype=np.int64)
        self.group_ids = np.array(self.group_ids, dtype=np.int64)
        self.norms = np.array(self.norms, dtype=np.float32)

        # ---------- 时间特征（占位，与 FIT 保持一致） ----------
        self.data_stamp_x = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        self.data_stamp_y = np.arange(
            self.label_len + self.pred_len
        ).reshape(-1, 1) / self.pred_len

        mode = "ElemOnly" if self.use_element else "NoMeta"
        log(f"[GeoStyle-{mode}][{self.flag}] loaded {len(self.series)} samples")
        log(f"  Elements: {len(self.element_map)} | Groups: {len(self.group_map)}")

    # ==================================================
    # Dataset API
    # ==================================================
    def __getitem__(self, index):
        """
        返回接口保持稳定：
        - element / group 是否使用由模型侧决定
        """

        seq_x = self.series[index].reshape(-1, 1)
        seq_y = self.targets[index].reshape(-1, 1)

        return (
            seq_x,
            seq_y,
            self.data_stamp_x,
            self.data_stamp_y,
            self.element_ids[index],
            self.group_ids[index],   # ← City / Group ID
            self.norms[index],       # ← [min, max, eps]
        )

    def __len__(self):
        return len(self.series)

    # GeoStyle 下不再支持 inverse_transform（保持接口但不使用）
    def inverse_transform(self, data):
        return data
