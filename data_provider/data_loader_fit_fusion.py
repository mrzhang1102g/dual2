"""
FIT fusion loader.

0417 active line:
- 仍然从 FIT 原始 json 读取数值序列与元数据
- 文本输入改为直接读取 json 中的原始字段（默认 `annotations`）
- 不再依赖离线 `caption_emb.pt`
"""

import numpy as np
from torch.utils.data import Dataset

from data_provider.fit_dataset_utils import load_fit_text_field, prepare_fit_dataset
from utils.logger import log


class Dataset_DualSG_Fit_Fusion(Dataset):
    """FIT 融合流数据集。"""

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
        text_field="annotations",
        seasonal_patterns=None,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        max_samples=-1,
        fit_scaler_mode="train_only",
    ):
        del timeenc, freq, seasonal_patterns, test_ratio
        assert flag in ["train", "val", "test"]

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.text_field = text_field
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.fit_scaler_mode = fit_scaler_mode

        if size is None:
            self.seq_len = 48
            self.label_len = 0
            self.pred_len = 24
        else:
            self.seq_len, self.label_len, self.pred_len = size

        prepared = prepare_fit_dataset(
            root_path=self.root_path,
            data_path=self.data_path,
            flag=self.flag,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            scale=self.scale,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            max_samples=max_samples,
            fit_scaler_mode=self.fit_scaler_mode,
        )

        self.series = prepared["series"]
        self.targets = prepared["targets"]
        self.city_ids = prepared["city_ids"]
        self.gender_ids = prepared["gender_ids"]
        self.age_ids = prepared["age_ids"]
        self.element_ids = prepared["element_ids"]
        self.element_map = prepared["element_map"]
        self.scaler = prepared["scaler"]
        self.fit_scaler_mode = prepared["fit_scaler_mode"]
        self.data_stamp_x = prepared["data_stamp_x"]
        self.data_stamp_y = prepared["data_stamp_y"]
        self.sample_indices = prepared["sample_indices"]

        self.num_cities = prepared["num_cities"]
        self.num_genders = prepared["num_genders"]
        self.num_ages = prepared["num_ages"]

        self.caption_text = load_fit_text_field(
            root_path=self.root_path,
            data_path=self.data_path,
            sample_indices=self.sample_indices,
            text_field=self.text_field,
        )

        log(f"[FIT-Fusion][{self.flag}] loaded {len(self.series)} samples")
        log(
            f"Elements:{len(self.element_map)}, "
            f"Cities:{self.num_cities}, "
            f"Genders:{self.num_genders}, "
            f"Ages:{self.num_ages}"
        )
        log(f"[FIT-Fusion][{self.flag}] scaler_mode={self.fit_scaler_mode}")
        log(f"[FIT-Fusion][{self.flag}] text_field={self.text_field}")

    def __getitem__(self, index):
        return (
            self.series[index].reshape(-1, 1),
            self.targets[index].reshape(-1, 1),
            self.data_stamp_x,
            self.data_stamp_y,
            self.city_ids[index],
            self.gender_ids[index],
            self.age_ids[index],
            self.element_ids[index],
            self.caption_text[index],
        )

    def __len__(self):
        return len(self.series)

    def inverse_transform(self, data):
        if not self.scale or self.scaler is None:
            return data
        array = np.asarray(data)
        return self.scaler.inverse_transform(array.reshape(-1, 1)).reshape(array.shape)
