"""FIT 数据集公共工具。

负责统一解析全量 json、构建元数据映射、执行 split，
并管理 caption embedding 与 scaler 的复用缓存。
"""

import json
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from utils.data_provider_utils import generate_time_features, is_all_mode, parse_fit_group, split_data


CITY_MAP = {
    "Cape Town": 0,
    "Hong Kong": 1,
    "London": 2,
    "Los Angeles": 3,
    "Milan": 4,
    "Moscow": 5,
    "New York": 6,
    "Paris": 7,
    "Rio de Janeiro": 8,
    "Seoul": 9,
    "Shanghai": 10,
    "Singapore": 11,
    "Sydney": 12,
    "Tokyo": 13,
}

GENDER_MAP = {"female": 0, "male": 1}

AGE_MAP = {
    "age_lt_18": 0,
    "age_18_25": 1,
    "age_25_40": 2,
    "age_gt_40": 3,
}


_FIT_JSON_CACHE = {}
_FIT_CAPTION_CACHE = {}
_FIT_SCALER_CACHE = {}


def _normalize_limit(max_samples):
    """把样本上限规整成 int；非正数表示不截断。"""
    if max_samples is None:
        return -1
    max_samples = int(max_samples)
    return max_samples if max_samples > 0 else -1


def load_fit_json(root_path, data_path):
    """读取 FIT 原始 json，并在进程内缓存。"""
    full_path = os.path.normpath(os.path.join(root_path, data_path))
    if full_path not in _FIT_JSON_CACHE:
        with open(full_path, "r", encoding="utf-8") as file:
            _FIT_JSON_CACHE[full_path] = json.load(file)
    return _FIT_JSON_CACHE[full_path]


def build_fit_element_map(full_data):
    """基于完整数据构建 element -> id 映射，保证各 split 一致。"""
    element_map = {}
    for item in full_data:
        element = item.get("metadata", {}).get("element", "unknown")
        if element not in element_map:
            element_map[element] = len(element_map)
    return element_map


def build_fit_sample_indices(total, data_path, flag, train_ratio, val_ratio, max_samples=-1):
    """先按正式 split 取索引，再应用本地 smoke 的样本上限。"""
    all_mode = is_all_mode(data_path)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    if all_mode:
        start, end, train_size, val_size = split_data(total, flag, train_ratio, val_ratio)
        sample_indices = list(range(start, end))
    else:
        sample_indices = list(range(total))

    max_samples = _normalize_limit(max_samples)
    if max_samples > 0:
        sample_indices = sample_indices[:max_samples]

    return {
        "all_mode": all_mode,
        "train_size": train_size,
        "val_size": val_size,
        "sample_indices": sample_indices,
    }


def _build_scaler_cache_key(root_path, data_path, train_ratio, val_ratio, fit_scaler_mode):
    full_path = os.path.normpath(os.path.join(root_path, data_path))
    return (full_path, float(train_ratio), float(val_ratio), str(fit_scaler_mode))


def _fit_scaler_from_series(series_array):
    scaler = StandardScaler()
    scaler.fit(series_array.reshape(-1, 1))
    return scaler


def _collect_series_from_indices(full_data, sample_indices):
    if not sample_indices:
        return np.empty((0,), dtype=np.float32)
    return np.asarray([full_data[index]["series"] for index in sample_indices], dtype=np.float32)


def _get_train_only_scaler(full_data, root_path, data_path, train_ratio, val_ratio):
    """只使用 train split 拟合 scaler，并缓存结果。"""
    cache_key = _build_scaler_cache_key(root_path, data_path, train_ratio, val_ratio, "train_only")
    if cache_key in _FIT_SCALER_CACHE:
        return _FIT_SCALER_CACHE[cache_key]

    if is_all_mode(data_path):
        train_size = int(len(full_data) * train_ratio)
        train_indices = list(range(train_size))
    else:
        # 非 all 模式下无法从单文件反推出独立训练集，只能退化为对当前文件拟合。
        train_indices = list(range(len(full_data)))

    train_series = _collect_series_from_indices(full_data, train_indices)
    scaler = _fit_scaler_from_series(train_series)
    _FIT_SCALER_CACHE[cache_key] = scaler
    return scaler


def _get_split_fit_scaler(series, root_path, data_path, train_ratio, val_ratio, flag):
    """保留旧行为：每个 split 各自拟合自己的 scaler。"""
    cache_key = (
        os.path.normpath(os.path.join(root_path, data_path)),
        float(train_ratio),
        float(val_ratio),
        "split_fit",
        str(flag),
    )
    if cache_key not in _FIT_SCALER_CACHE:
        _FIT_SCALER_CACHE[cache_key] = _fit_scaler_from_series(series)
    return _FIT_SCALER_CACHE[cache_key]


def _select_fit_scaler(full_data, series, root_path, data_path, flag, train_ratio, val_ratio, fit_scaler_mode):
    if fit_scaler_mode == "train_only":
        return _get_train_only_scaler(full_data, root_path, data_path, train_ratio, val_ratio)
    if fit_scaler_mode == "split_fit":
        return _get_split_fit_scaler(series, root_path, data_path, train_ratio, val_ratio, flag)
    raise ValueError(f"Unsupported fit_scaler_mode: {fit_scaler_mode}")


def prepare_fit_dataset(
    root_path,
    data_path,
    flag,
    seq_len,
    label_len,
    pred_len,
    scale,
    train_ratio,
    val_ratio,
    max_samples=-1,
    fit_scaler_mode="train_only",
):
    """一次性准备 FIT loader 需要的公共字段。"""
    full_data = load_fit_json(root_path, data_path)
    split_info = build_fit_sample_indices(
        total=len(full_data),
        data_path=data_path,
        flag=flag,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_samples=max_samples,
    )

    sample_indices = split_info["sample_indices"]
    element_map = build_fit_element_map(full_data)

    series = []
    targets = []
    city_ids = []
    gender_ids = []
    age_ids = []
    element_ids = []

    for index in sample_indices:
        item = full_data[index]
        metadata = item.get("metadata", {})
        city, gender, age = parse_fit_group(metadata.get("group", ""))
        element = metadata.get("element", "unknown")

        series.append(item["series"])
        targets.append(item["target"])
        city_ids.append(CITY_MAP.get(city, 0))
        gender_ids.append(GENDER_MAP.get(gender, 0))
        age_ids.append(AGE_MAP.get(age, 0))
        element_ids.append(element_map.get(element, 0))

    series = np.asarray(series, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    city_ids = np.asarray(city_ids, dtype=np.int64)
    gender_ids = np.asarray(gender_ids, dtype=np.int64)
    age_ids = np.asarray(age_ids, dtype=np.int64)
    element_ids = np.asarray(element_ids, dtype=np.int64)

    scaler = None
    if scale and len(series) > 0:
        scaler = _select_fit_scaler(
            full_data=full_data,
            series=series,
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            fit_scaler_mode=fit_scaler_mode,
        )
        series = scaler.transform(series.reshape(-1, 1)).reshape(series.shape)
        targets = scaler.transform(targets.reshape(-1, 1)).reshape(targets.shape)

    data_stamp_x, data_stamp_y = generate_time_features(seq_len, label_len, pred_len)

    return {
        "series": series,
        "targets": targets,
        "city_ids": city_ids,
        "gender_ids": gender_ids,
        "age_ids": age_ids,
        "element_ids": element_ids,
        "element_map": element_map,
        "sample_indices": sample_indices,
        "all_mode": split_info["all_mode"],
        "train_size": split_info["train_size"],
        "val_size": split_info["val_size"],
        "total": len(full_data),
        "num_cities": len(CITY_MAP),
        "num_genders": len(GENDER_MAP),
        "num_ages": len(AGE_MAP),
        "scaler": scaler,
        "fit_scaler_mode": fit_scaler_mode,
        "data_stamp_x": data_stamp_x,
        "data_stamp_y": data_stamp_y,
    }


def load_fit_caption_embeddings(caption_emb_path, sample_indices):
    """按 `sample_indices` 对齐离线文本 embedding。"""
    caption_emb_path = os.path.normpath(caption_emb_path)
    if caption_emb_path not in _FIT_CAPTION_CACHE:
        full_emb = torch.load(caption_emb_path, map_location="cpu")
        if not isinstance(full_emb, torch.Tensor):
            full_emb = torch.tensor(full_emb)
        _FIT_CAPTION_CACHE[caption_emb_path] = full_emb

    full_emb = _FIT_CAPTION_CACHE[caption_emb_path]
    if full_emb.ndim != 2:
        raise ValueError(f"caption_emb must be 2D tensor [N, D], got shape={tuple(full_emb.shape)}")

    max_index = max(sample_indices) if sample_indices else -1
    if full_emb.shape[0] <= max_index:
        raise ValueError(
            f"caption_emb rows={full_emb.shape[0]} <= required max index={max_index}. "
            "请确认 embedding 与原始 json 使用同一顺序。"
        )

    return full_emb[sample_indices].float()
