"""Geo 融合流数据集。

目标是与 Geo 数值流保持同一套切分和元数据映射，
只在返回值末尾额外追加 `caption_emb`。
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_provider_utils import build_metadata_maps, generate_time_features, is_all_mode, split_data
from utils.logger import log


class Dataset_DualSG_Geo_Fusion(Dataset):
    """GeoStyle 融合流数据集。"""

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
        caption_emb_path=None,
        seasonal_patterns=None,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        use_element=False,
    ):
        del scale, timeenc, freq, seasonal_patterns, test_ratio
        assert flag in ["train", "val", "test"]
        if caption_emb_path is None:
            raise ValueError("caption_emb_path must be provided for Geo_Fusion")

        self.flag = flag
        self.features = features
        self.target = target
        self.use_element = use_element
        self.root_path = root_path
        self.data_path = data_path
        self.caption_emb_path = caption_emb_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        if size is None:
            self.seq_len = 52
            self.label_len = 0
            self.pred_len = 26
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.element_map = {}
        self.group_map = {}
        self.sample_indices = None

        self._read_data()
        self._load_caption_embedding()

    def _read_data(self):
        full_path = os.path.join(self.root_path, self.data_path)
        with open(full_path, "r", encoding="utf-8") as file:
            full_data = json.load(file)

        total = len(full_data)
        if is_all_mode(self.data_path):
            start, end, _, _ = split_data(total, self.flag, self.train_ratio, self.val_ratio)
            data_list = full_data[start:end]
            self.sample_indices = list(range(start, end))
        else:
            data_list = full_data
            self.sample_indices = list(range(total))

        self.element_map, self.group_map = build_metadata_maps(full_data)
        if len(self.element_map) == 0 or len(self.group_map) == 0:
            raise RuntimeError("element_map/group_map build failed: mapping is empty")

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
        log(f"[GeoStyle-Fusion-{mode}][{self.flag}] loaded {len(self.series)} samples")
        log(f"  Elements: {len(self.element_map)} | Groups: {len(self.group_map)}")

        if is_all_mode(self.data_path):
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)
            test_size = total - train_size - val_size
            log(f"  Split(all): total={total}, train={train_size}, val={val_size}, test={test_size}")

    def _load_caption_embedding(self):
        """按原始 json 顺序对齐 caption embedding。"""
        log(f"[GeoStyle-Fusion] Loading caption embedding from: {self.caption_emb_path}")
        full_emb = torch.load(self.caption_emb_path, map_location="cpu")

        if not isinstance(full_emb, torch.Tensor):
            full_emb = torch.tensor(full_emb)
        if full_emb.ndim != 2:
            raise ValueError(f"caption_emb must be 2D tensor [N, D], got shape={tuple(full_emb.shape)}")

        max_idx = max(self.sample_indices) if self.sample_indices else -1
        if full_emb.shape[0] <= max_idx:
            raise ValueError(
                f"caption_emb rows={full_emb.shape[0]} <= required max index={max_idx}. "
                "Check whether caption_emb is generated from the same json."
            )

        self.caption_emb = full_emb[self.sample_indices].float()
        if len(self.caption_emb) != len(self.series):
            raise RuntimeError(
                f"caption_emb length mismatch: caption_emb={len(self.caption_emb)} vs series={len(self.series)}. "
                "Your sample_indices slicing or json ordering is inconsistent."
            )

        log(f"[GeoStyle-Fusion] caption_emb aligned: {tuple(self.caption_emb.shape)}")

    def __getitem__(self, index):
        return (
            self.series[index].reshape(-1, 1),
            self.targets[index].reshape(-1, 1),
            self.data_stamp_x,
            self.data_stamp_y,
            self.element_ids[index],
            self.group_ids[index],
            self.norms[index],
            self.caption_emb[index],
        )

    def __len__(self):
        return len(self.series)

    def inverse_transform(self, data):
        return data
