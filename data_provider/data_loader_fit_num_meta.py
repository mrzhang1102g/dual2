"""
FIT 数值流 + 元数据 loader。

说明：
- 纯数值模型和数值+meta 模型目前都复用这一个 dataset。
- dataset 始终返回完整的 FIT 元数据四元组，是否使用由模型侧决定。
"""

import numpy as np
from torch.utils.data import Dataset

from data_provider.fit_dataset_utils import prepare_fit_dataset
from utils.logger import log


class Dataset_DualSG_Fit_Num_Meta(Dataset):
    """FIT 数值流数据集。"""

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
        test_ratio=0.2,
        max_samples=-1,
        fit_scaler_mode="train_only",
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

        self.num_cities = prepared["num_cities"]
        self.num_genders = prepared["num_genders"]
        self.num_ages = prepared["num_ages"]

        log(f"[FIT-Meta][{self.flag}] loaded {len(self.series)} samples")
        log(
            f"Elements:{len(self.element_map)}, "
            f"Cities:{self.num_cities}, "
            f"Genders:{self.num_genders}, "
            f"Ages:{self.num_ages}"
        )
        log(f"[FIT-Meta][{self.flag}] scaler_mode={self.fit_scaler_mode}")

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
        )

    def __len__(self):
        return len(self.series)

    def inverse_transform(self, data):
        if not self.scale or self.scaler is None:
            return data
        return self.scaler.inverse_transform(np.asarray(data).reshape(-1, 1)).reshape(np.asarray(data).shape)
