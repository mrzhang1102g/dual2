"""
DualSG Fusion Loader (GeoStyle)
============================================

目标：让融合流在“数据行为”上与数值流(Dataset_DualSG_Geo_Num_Meta)严格对齐，
但保留融合流额外的 caption_emb 输出。

对齐点（你提到的那些问题）：
1) split：与数值流一致（all模式下固定 70/10/20），不再允许自由 split_ratio 造成偏移
2) data_stamp_y：与数值流一致，使用 (label_len + pred_len)
3) caption_emb：强对齐 + 强校验（长度一致、索引一致）
4) element/group 映射：强制基于 full_data 构建（避免未来改坏）
5) 返回 tuple：前7项与数值流完全一致，仅在最后追加 caption_emb

返回：
(
  seq_x, seq_y,
  seq_x_mark, seq_y_mark,
  element_id,
  group_id,
  norm,
  caption_emb
)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.logger import log


# ======================================================
# 工具函数
# ======================================================
from utils.data_provider_utils import is_all_mode, split_data, generate_time_features, build_metadata_maps


# ======================================================
# Dataset
# ======================================================
class Dataset_DualSG_Geo_Fusion(Dataset):
    """
    GeoStyle Fusion Dataset (DualSG-compatible, KERN-aligned)

    说明：
    - 不改 GeoStyle 数值本身
    - dataloader 逻辑与数值流对齐：切分、时间戳、元数据映射来源
    - caption_emb 只做对齐与返回，不影响数值流行为
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="geo_dualsg_all.json",
        target="trend",
        scale=True,  # GeoStyle 下保留参数但不使用，保持接口兼容
        timeenc=0,
        freq="w",
        caption_emb_path=None,
        seasonal_patterns=None,
        train_ratio=0.7,  # 训练数据比例
        val_ratio=0.1,    # 验证数据比例
        test_ratio=0.2,   # 测试数据比例
        use_element=False,
    ):
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
        self.test_ratio = test_ratio

        # GeoStyle 序列长度
        if size is None:
            # 使用配置的默认值
            self.seq_len = 52
            self.label_len = 0
            self.pred_len = 26
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # ---- 元数据映射表（必须基于 full_data 全局构建）----
        self.element_map = {}
        self.group_map = {}

        # ---- all 模式标记 & 对齐索引 ----
        self.is_all_mode = is_all_mode(self.data_path)
        self.sample_indices = None  # 用于 caption_emb 强对齐

        self.__read_data__()
        self.__load_caption_embedding__()

    # ==================================================
    # 读取并处理数据
    # ==================================================
    def __read_data__(self):
        # ---------- 1) 读取 JSON ----------
        full_path = os.path.join(self.root_path, self.data_path)
        with open(full_path, "r") as f:
            full_data = json.load(f)

        total = len(full_data)

        # ---------- 2) 与数值流一致的切分 ----------
        if self.is_all_mode:
            start, end, train_size, val_size = split_data(total, self.flag, self.train_ratio, self.val_ratio)
            data_list = full_data[start:end]
            self.sample_indices = list(range(start, end))
        else:
            data_list = full_data
            self.sample_indices = list(range(total))

        # ---------- 3) element/group 映射必须基于 full_data 构建 ----------
        self.element_map, self.group_map = build_metadata_maps(full_data)

        # 防止未来维护时把映射构建逻辑改坏
        if len(self.element_map) == 0 or len(self.group_map) == 0:
            raise RuntimeError("element_map/group_map build failed: mapping is empty")

        # ---------- 4) 提取样本 ----------
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

            element = meta.get("element", "unknown")
            group = meta.get("group", "unknown")
            norm = meta.get("norm", [0.0, 1.0, 0.0])

            self.element_ids.append(self.element_map.get(element, 0))
            self.group_ids.append(self.group_map.get(group, 0))
            self.norms.append(norm)

        # ---------- 5) numpy ----------
        self.series = np.array(self.series, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.element_ids = np.array(self.element_ids, dtype=np.int64)
        self.group_ids = np.array(self.group_ids, dtype=np.int64)
        self.norms = np.array(self.norms, dtype=np.float32)

        # ---------- 6) 时间特征：与数值流严格对齐 ----------
        self.data_stamp_x, self.data_stamp_y = generate_time_features(self.seq_len, self.label_len, self.pred_len)

        mode = "ElemOnly" if self.use_element else "NoMeta"
        log(f"[GeoStyle-Fusion-{mode}][{self.flag}] loaded {len(self.series)} samples")
        log(f"  Elements: {len(self.element_map)} | Groups: {len(self.group_map)}")
        if self.is_all_mode:
            # 仅用于日志，确认切分行为与 Num 一致
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)
            test_size = total - train_size - val_size
            log(f"  Split(all): total={total}, train={train_size}, val={val_size}, test={test_size}")

    # ==================================================
    # caption embedding 强对齐加载
    # ==================================================
    def __load_caption_embedding__(self):
        """
        caption_emb 必须与 json 的“原始索引”对齐：
        - caption_emb_all.pt 的第 i 行对应 geo_dualsg_all.json 的第 i 个样本
        """
        log(f"[GeoStyle-Fusion] Loading caption embedding from: {self.caption_emb_path}")
        full_emb = torch.load(self.caption_emb_path)

        if not isinstance(full_emb, torch.Tensor):
            full_emb = torch.tensor(full_emb)

        # 1) 基础形状校验
        if full_emb.ndim != 2:
            raise ValueError(f"caption_emb must be 2D tensor [N,D], got shape={tuple(full_emb.shape)}")

        # 2) 索引覆盖校验
        max_idx = max(self.sample_indices) if len(self.sample_indices) > 0 else -1
        if full_emb.shape[0] <= max_idx:
            raise ValueError(
                f"caption_emb rows={full_emb.shape[0]} <= required max index={max_idx}. "
                f"Check whether caption_emb is generated from the same json."
            )

        # 3) 对齐切片
        self.caption_emb = full_emb[self.sample_indices].float()

        # 4) 强一致性校验
        if len(self.caption_emb) != len(self.series):
            raise RuntimeError(
                f"caption_emb length mismatch: caption_emb={len(self.caption_emb)} vs series={len(self.series)}. "
                f"Your sample_indices slicing or json ordering is inconsistent."
            )

        log(f"[GeoStyle-Fusion] caption_emb aligned: {tuple(self.caption_emb.shape)}")

    # ==================================================
    # Dataset API
    # ==================================================
    def __getitem__(self, index):
        """
            在最后追加 caption_emb。
        """
        seq_x = self.series[index].reshape(-1, 1)
        seq_y = self.targets[index].reshape(-1, 1)

        return (
            seq_x,
            seq_y,
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
        # GeoStyle 不做 scaler，保持接口兼容
        return data
