"""
Shared FIT dataset utilities.

Responsibilities:
- unify FIT numeric / fusion loader parsing logic
- provide optional local smoke sample limits
- manage FIT scaling strategy
- parse structured annotations into semantic supervision labels
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

TREND_LABEL_MAP = {
    "falling": 0,
    "stable": 1,
    "rising": 2,
}

VOLATILITY_LABEL_MAP = {
    "low": 0,
    "moderate": 1,
    "high": 2,
}

TURNING_LABEL_MAP = {
    "none": 0,
    "early": 1,
    "middle": 2,
    "late": 3,
}


_FIT_JSON_CACHE = {}
_FIT_CAPTION_CACHE = {}
_FIT_SCALER_CACHE = {}


def _normalize_limit(max_samples):
    if max_samples is None:
        return -1
    max_samples = int(max_samples)
    return max_samples if max_samples > 0 else -1


def load_fit_json(root_path, data_path):
    """
    Read FIT raw json and cache it per process.
    """
    full_path = os.path.normpath(os.path.join(root_path, data_path))
    if full_path not in _FIT_JSON_CACHE:
        with open(full_path, "r", encoding="utf-8") as file:
            _FIT_JSON_CACHE[full_path] = json.load(file)
    return _FIT_JSON_CACHE[full_path]


def build_fit_element_map(full_data):
    """
    Build a stable element -> id mapping from the full dataset.
    """
    element_map = {}
    for item in full_data:
        element = item.get("metadata", {}).get("element", "unknown")
        if element not in element_map:
            element_map[element] = len(element_map)
    return element_map


def build_fit_sample_indices(total, data_path, flag, train_ratio, val_ratio, max_samples=-1):
    """
    Split first, then optionally truncate for local smoke runs.
    """
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
    """
    Fit scaler on train split only and reuse it for val/test.
    """
    cache_key = _build_scaler_cache_key(root_path, data_path, train_ratio, val_ratio, "train_only")
    if cache_key in _FIT_SCALER_CACHE:
        return _FIT_SCALER_CACHE[cache_key]

    if is_all_mode(data_path):
        train_size = int(len(full_data) * train_ratio)
        train_indices = list(range(train_size))
    else:
        train_indices = list(range(len(full_data)))

    train_series = _collect_series_from_indices(full_data, train_indices)
    scaler = _fit_scaler_from_series(train_series)
    _FIT_SCALER_CACHE[cache_key] = scaler
    return scaler


def _get_split_fit_scaler(series, root_path, data_path, train_ratio, val_ratio, flag):
    """
    Compatibility mode: each split fits its own scaler.
    """
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
    """
    Prepare the common FIT fields used by loaders.
    """
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
    """
    Align offline caption embeddings with sample_indices.
    """
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
            "Please make sure the embedding rows follow the same order as the source json."
        )

    return full_emb[sample_indices].float()


def load_fit_text_field(root_path, data_path, sample_indices, text_field="annotations"):
    """
    Align raw FIT text fields with sample_indices.
    """
    full_data = load_fit_json(root_path, data_path)
    texts = []
    for index in sample_indices:
        item = full_data[index]
        value = item.get(text_field, "")
        texts.append("" if value is None else str(value))
    return texts


def _safe_json_loads(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, dict) else {"caption": str(loaded)}
        except json.JSONDecodeError:
            return {"caption": text}
    return {"caption": str(value)}


def _normalize_text(value):
    return str(value or "").strip().lower()


def _classify_trend(value):
    text = _normalize_text(value)
    if not text:
        return None

    falling_keywords = ("fall", "falling", "drop", "declin", "declining", "down", "decreas", "weaken")
    rising_keywords = (
        "rise",
        "rising",
        "grow",
        "growing",
        "increase",
        "increasing",
        "climb",
        "surge",
        "strengthen",
        "upward",
    )
    stable_keywords = ("stable", "flat", "steady", "plateau", "sideway", "unchanged", "balanced")

    if any(keyword in text for keyword in falling_keywords):
        return TREND_LABEL_MAP["falling"]
    if any(keyword in text for keyword in rising_keywords):
        return TREND_LABEL_MAP["rising"]
    if any(keyword in text for keyword in stable_keywords):
        return TREND_LABEL_MAP["stable"]
    return None


def _classify_volatility(value):
    text = _normalize_text(value)
    if not text:
        return None

    low_keywords = ("low", "calm", "smooth", "mild")
    moderate_keywords = ("moderate", "medium", "mixed")
    high_keywords = ("high", "volatile", "sharp", "strong", "intense", "unstable")

    if any(keyword in text for keyword in low_keywords):
        return VOLATILITY_LABEL_MAP["low"]
    if any(keyword in text for keyword in high_keywords):
        return VOLATILITY_LABEL_MAP["high"]
    if any(keyword in text for keyword in moderate_keywords):
        return VOLATILITY_LABEL_MAP["moderate"]
    return None


def _classify_turning(value):
    if value is None:
        return TURNING_LABEL_MAP["none"], 0.0

    if isinstance(value, list):
        if not value:
            return TURNING_LABEL_MAP["none"], 1.0
        joined = " ".join(str(item) for item in value)
    elif isinstance(value, dict):
        joined = " ".join(f"{key} {item}" for key, item in value.items())
    else:
        joined = str(value)

    text = _normalize_text(joined)
    if not text:
        return TURNING_LABEL_MAP["none"], 0.0
    if any(keyword in text for keyword in ("none", "no turning", "no major turning")):
        return TURNING_LABEL_MAP["none"], 1.0
    if "early" in text:
        return TURNING_LABEL_MAP["early"], 1.0
    if "middle" in text or "mid " in text or "mid-" in text:
        return TURNING_LABEL_MAP["middle"], 1.0
    if "late" in text:
        return TURNING_LABEL_MAP["late"], 1.0

    # Turning points exist but the stage is not explicit.
    return TURNING_LABEL_MAP["middle"], 1.0


def parse_fit_semantic_targets(annotation_value):
    """
    Parse structured annotations into four semantic supervision targets.

    The parser is intentionally robust enough to handle:
    - full structured json with multiple fields
    - split view json files with only a subset of fields
    - plain caption strings as a fallback
    """
    annotation = _safe_json_loads(annotation_value)
    caption = annotation.get("caption", "")

    overall_label = _classify_trend(annotation.get("overall_trend", caption))
    recent_label = _classify_trend(
        annotation.get("recent_regime", annotation.get("recent_trend", caption))
    )
    volatility_label = _classify_volatility(annotation.get("volatility", caption))
    turning_label, turning_mask = _classify_turning(annotation.get("major_turning_points"))

    return {
        "overall_label": int(overall_label if overall_label is not None else TREND_LABEL_MAP["stable"]),
        "recent_label": int(recent_label if recent_label is not None else TREND_LABEL_MAP["stable"]),
        "volatility_label": int(
            volatility_label if volatility_label is not None else VOLATILITY_LABEL_MAP["moderate"]
        ),
        "turning_label": int(turning_label),
        "overall_mask": float(overall_label is not None),
        "recent_mask": float(recent_label is not None),
        "volatility_mask": float(volatility_label is not None),
        "turning_mask": float(turning_mask),
    }


def load_fit_semantic_targets(root_path, data_path, sample_indices, text_field="annotations"):
    """
    Align structured annotations with sample_indices and convert them to semantic targets.
    """
    full_data = load_fit_json(root_path, data_path)
    targets = []
    for index in sample_indices:
        item = full_data[index]
        annotation_value = item.get(text_field, "")
        targets.append(parse_fit_semantic_targets(annotation_value))
    return targets
