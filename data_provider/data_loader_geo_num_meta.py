"""Geo 数值流数据集。"""

import json
import os

import numpy as np
from torch.utils.data import Dataset

from utils.data_provider_utils import build_metadata_maps, generate_time_features, is_all_mode, split_data
from utils.logger import log


class Dataset_DualSG_Geo_Num_Meta(Dataset):
    """GeoStyle 数值流数据集。"""

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
        max_samples=-1,
    ):
        del scale, timeenc, freq, seasonal_patterns, test_ratio
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.use_element = use_element
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.max_samples = max_samples

        if size is None:
            self.seq_len = 52
            self.label_len = 0
            self.pred_len = 26
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.element_map = {}
        self.group_map = {}
        self._read_data()

    def _read_data(self):
        full_path = os.path.join(self.root_path, self.data_path)
        with open(full_path, "r", encoding="utf-8") as file:
            full_data = json.load(file)

        if is_all_mode(self.data_path):
            total = len(full_data)
            start, end, _, _ = split_data(total, self.flag, self.train_ratio, self.val_ratio)
            data_list = full_data[start:end]
        else:
            data_list = full_data

        if self.max_samples is not None and self.max_samples > 0:
            data_list = data_list[: self.max_samples]

        self.element_map, self.group_map = build_metadata_maps(full_data)

        self.series = []
        self.targets = []
        self.element_ids = []
        self.group_ids = []
        self.norms = []

        for item in data_list:
            meta = item.get("metadata", {})
            self.series.append(item["series"])
            self.targets.append(item["target"])

            element = meta.get("element", "unknown")
            group = meta.get("group", "unknown")
            norm = meta.get("norm", [0.0, 1.0, 0.0])

            self.element_ids.append(self.element_map.get(element, 0))
            self.group_ids.append(self.group_map.get(group, 0))
            self.norms.append(norm)

        self.series = np.asarray(self.series, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)
        self.element_ids = np.asarray(self.element_ids, dtype=np.int64)
        self.group_ids = np.asarray(self.group_ids, dtype=np.int64)
        self.norms = np.asarray(self.norms, dtype=np.float32)
        self.data_stamp_x, self.data_stamp_y = generate_time_features(self.seq_len, self.label_len, self.pred_len)

        mode = "ElemOnly" if self.use_element else "NoMeta"
        log(f"[GeoStyle-{mode}][{self.flag}] loaded {len(self.series)} samples")
        log(f"  Elements: {len(self.element_map)} | Groups: {len(self.group_map)}")

    def __getitem__(self, index):
        return (
            self.series[index].reshape(-1, 1),
            self.targets[index].reshape(-1, 1),
            self.data_stamp_x,
            self.data_stamp_y,
            self.element_ids[index],
            self.group_ids[index],
            self.norms[index],
        )

    def __len__(self):
        return len(self.series)

    def inverse_transform(self, data):
        # GeoStyle 不使用全局 scaler，这里仅保留接口一致性。
        return data
