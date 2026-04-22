"""FIT 融合流数据集。

先从 FIT json 读取数值序列和元数据，再按同一组 sample indices
对齐离线生成的 caption embedding。
"""

import numpy as np
from torch.utils.data import Dataset

from data_provider.fit_dataset_utils import load_fit_caption_embeddings, prepare_fit_dataset
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
        caption_emb_path=None,
        seasonal_patterns=None,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        max_samples=-1,
        fit_scaler_mode="train_only",
    ):
        del timeenc, freq, seasonal_patterns, test_ratio
        assert flag in ["train", "val", "test"]
        if caption_emb_path is None:
            raise ValueError("caption_emb_path must be provided for FIT_Fusion")

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.caption_emb_path = caption_emb_path
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

        log(f"[FIT-Fusion][{self.flag}] loaded {len(self.series)} samples")
        log(
            f"Elements:{len(self.element_map)}, "
            f"Cities:{self.num_cities}, "
            f"Genders:{self.num_genders}, "
            f"Ages:{self.num_ages}"
        )
        log(f"[FIT-Fusion][{self.flag}] scaler_mode={self.fit_scaler_mode}")

        # caption embedding 必须与当前 split 的 sample_indices 严格对齐。
        log(f"[FIT-Fusion] Loading caption embedding from: {self.caption_emb_path}")
        self.caption_emb = load_fit_caption_embeddings(self.caption_emb_path, self.sample_indices)
        log(f"[FIT-Fusion] caption_emb aligned: {tuple(self.caption_emb.shape)}")

    def __getitem__(self, index):
        # 末尾追加 caption_emb，前 8 项的顺序与数值流保持稳定。
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
        return len(self.series)

    def inverse_transform(self, data):
        # FIT 在可视化和保存结果时走统一反归一化入口。
        if not self.scale or self.scaler is None:
            return data
        array = np.asarray(data)
        return self.scaler.inverse_transform(array.reshape(-1, 1)).reshape(array.shape)
