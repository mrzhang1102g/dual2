#!/usr/bin/env python3
"""
Generate metadata-aware English annotations for FIT_DualSG with LangChain.

Route-2 input:
- historical series
- metadata (city / gender / age group / element)

Design choices:
- plain-text output only, no structured parsing
- JSONL progress file for resume
- merged JSON checkpoint synced every N samples
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

AGE_LABEL_MAP = {
    "age_lt_18": "under 18",
    "age_18_25": "18 to 25",
    "age_25_40": "25 to 40",
    "age_gt_40": "over 40",
}

SYSTEM_PROMPT_EN = """You write concise English annotations for fashion trend time series.

Use the metadata as domain context, but base every claim on the provided historical series only.
Do not predict future target values.
Do not mention model training, prompts, hidden features, or uncertainty boilerplate.
Return exactly one short paragraph in natural English.
No bullet points, no JSON, no markdown, no title.
"""

# 中文翻译，仅供人工对照，不参与实际调用。
SYSTEM_PROMPT_ZH = """你要为时尚趋势时间序列写简洁的英文描述。

可以把元数据当作领域上下文，但所有判断都必须基于给定的历史序列本身。
不要预测未来 target。
不要提模型训练、提示词、隐藏特征，也不要写空泛免责声明。
只返回一段简短自然的英文段落。
不要项目符号，不要 JSON，不要 Markdown，不要标题。
"""

USER_PROMPT_EN = """Metadata
- Element: {element}
- City: {city}
- Gender: {gender}
- Age group: {age_group}

Historical series
- Length: {series_length}
- Values: [{series_values}]

Numerical summary
- Min / max: {min_value} / {max_value}
- Mean / std: {mean_value} / {std_value}
- First / last: {first_value} / {last_value}
- Overall change: {overall_change}
- Recent change: {recent_change}
- Overall trend tag: {overall_trend}
- Recent trend tag: {recent_trend}
- Volatility tag: {volatility_tag}
- Peak position: {peak_position} (index {peak_index}, value {peak_value})
- Trough position: {trough_position} (index {trough_index}, value {trough_value})
- Largest rise: {largest_rise} around step {largest_rise_step}
- Largest drop: {largest_drop} around step {largest_drop_step}

Write one concise English paragraph for this historical fashion trend series.

Requirements
1. Use the metadata as meaningful context, not as filler.
2. Describe the observed historical pattern only.
3. Mention the overall pattern, volatility, and recent behavior.
4. Keep it specific and fluent, not generic.
5. Prefer 2 to 4 sentences and roughly 45 to 110 words.
6. Return only the paragraph text.
"""

# 中文翻译，仅供人工对照，不参与实际调用。
USER_PROMPT_ZH = """输入信息
- Element: {element}
- City: {city}
- Gender: {gender}
- Age group: {age_group}

历史序列
- 长度: {series_length}
- 数值: [{series_values}]

数值摘要
- 最小 / 最大: {min_value} / {max_value}
- 均值 / 标准差: {mean_value} / {std_value}
- 起点 / 终点: {first_value} / {last_value}
- 整体变化: {overall_change}
- 近期变化: {recent_change}
- 整体趋势标签: {overall_trend}
- 近期趋势标签: {recent_trend}
- 波动标签: {volatility_tag}
- 峰值位置: {peak_position} (index {peak_index}, value {peak_value})
- 谷值位置: {trough_position} (index {trough_index}, value {trough_value})
- 最大上升: {largest_rise} around step {largest_rise_step}
- 最大下降: {largest_drop} around step {largest_drop_step}

请为这条时尚趋势历史序列写一段简洁英文描述。

要求
1. 元数据要真正作为上下文，而不是填充废话。
2. 只描述已观察到的历史模式。
3. 需要提到整体模式、波动情况和近期行为。
4. 表达要具体、流畅，不要泛泛而谈。
5. 优先写成 2 到 4 句，约 45 到 110 个英文词。
6. 只返回段落文本。
"""


@dataclass
class RunConfig:
    input_json_path: Path
    output_dir: Path
    output_json_filename: str
    progress_jsonl_filename: str
    error_jsonl_filename: str
    start_index: int = 0
    end_index: int | None = None  # exclusive
    flush_every: int = 10
    max_concurrency: int = 5
    max_retries: int = 5
    retry_backoff_seconds: float = 3.0
    request_timeout_seconds: float = 120.0
    temperature: float = 0.2
    max_tokens: int = 180
    model_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    sync_merged_json_on_flush: bool = True

    @property
    def output_json_path(self) -> Path:
        return self.output_dir / self.output_json_filename

    @property
    def progress_jsonl_path(self) -> Path:
        return self.output_dir / self.progress_jsonl_filename

    @property
    def error_jsonl_path(self) -> Path:
        return self.output_dir / self.error_jsonl_filename


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_group(group: str) -> dict[str, str]:
    parsed = {"age": "unknown", "city": "unknown", "gender": "unknown"}
    if not group:
        return parsed

    for chunk in group.split("__"):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        parsed[key.strip()] = value.strip()

    return parsed


def prettify_age_group(age_group: str) -> str:
    return AGE_LABEL_MAP.get(age_group, age_group or "unknown")


def format_float(value: float) -> str:
    return f"{value:.4f}"


def position_label(index: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    ratio = index / max(total - 1, 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "middle"
    return "late"


def classify_trend(delta: float, span: float) -> str:
    threshold = max(span * 0.15, 1e-4)
    if delta > threshold:
        return "rising"
    if delta < -threshold:
        return "falling"
    return "roughly stable"


def classify_volatility(std_value: float, mean_abs_diff: float, span: float) -> str:
    base = max(span, 1e-4)
    score = max(std_value / base, mean_abs_diff / base)
    if score < 0.15:
        return "low"
    if score < 0.3:
        return "moderate"
    return "high"


def summarize_series(series: list[float]) -> dict[str, Any]:
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("series must be a non-empty 1D list")

    diffs = np.diff(arr)
    span = float(arr.max() - arr.min())
    recent_window = min(8, arr.size)
    prev_window = arr[-2 * recent_window : -recent_window] if arr.size >= recent_window * 2 else arr[:-recent_window]

    recent_mean = float(arr[-recent_window:].mean())
    prev_mean = float(prev_window.mean()) if prev_window.size > 0 else float(arr[0])
    recent_change = recent_mean - prev_mean

    overall_change = float(arr[-1] - arr[0])
    mean_abs_diff = float(np.mean(np.abs(diffs))) if diffs.size > 0 else 0.0

    largest_rise_idx = int(np.argmax(diffs) + 1) if diffs.size > 0 else 0
    largest_drop_idx = int(np.argmin(diffs) + 1) if diffs.size > 0 else 0
    largest_rise = float(diffs[largest_rise_idx - 1]) if diffs.size > 0 else 0.0
    largest_drop = float(diffs[largest_drop_idx - 1]) if diffs.size > 0 else 0.0

    peak_idx = int(np.argmax(arr))
    trough_idx = int(np.argmin(arr))

    return {
        "series_length": int(arr.size),
        "series_values": ", ".join(format_float(v) for v in arr.tolist()),
        "min_value": format_float(float(arr.min())),
        "max_value": format_float(float(arr.max())),
        "mean_value": format_float(float(arr.mean())),
        "std_value": format_float(float(arr.std(ddof=0))),
        "first_value": format_float(float(arr[0])),
        "last_value": format_float(float(arr[-1])),
        "overall_change": format_float(overall_change),
        "recent_change": format_float(recent_change),
        "overall_trend": classify_trend(overall_change, span),
        "recent_trend": classify_trend(recent_change, span),
        "volatility_tag": classify_volatility(float(arr.std(ddof=0)), mean_abs_diff, span),
        "peak_index": peak_idx,
        "peak_position": position_label(peak_idx, int(arr.size)),
        "peak_value": format_float(float(arr[peak_idx])),
        "trough_index": trough_idx,
        "trough_position": position_label(trough_idx, int(arr.size)),
        "trough_value": format_float(float(arr[trough_idx])),
        "largest_rise_step": largest_rise_idx,
        "largest_rise": format_float(largest_rise),
        "largest_drop_step": largest_drop_idx,
        "largest_drop": format_float(largest_drop),
    }


def build_prompt_payload(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    group_info = parse_group(str(metadata.get("group", "")))
    summary = summarize_series(item["series"])

    payload = {
        "element": metadata.get("element", "unknown"),
        "city": group_info.get("city", "unknown"),
        "gender": group_info.get("gender", "unknown"),
        "age_group": prettify_age_group(group_info.get("age", "unknown")),
    }
    payload.update(summary)
    return payload


def clean_generated_text(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    for prefix in ("Description:", "Annotation:", "Output:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()

    cleaned = cleaned.strip().strip('"').strip("'").strip()
    cleaned = " ".join(cleaned.split())
    return cleaned


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list from {path}, got {type(data).__name__}")
    return data


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_progress(path: Path) -> dict[int, str]:
    progress: dict[int, str] = {}
    if not path.exists():
        return progress

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = int(record["index"])
            progress[index] = str(record["text"])

    return progress


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


class MetadataAwareAnnotationGenerator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_EN),
                ("human", USER_PROMPT_EN),
            ]
        )
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout_seconds,
            max_retries=0,
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def generate_one(
        self,
        index: int,
        item: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, str | None, str | None]:
        payload = build_prompt_payload(item)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with semaphore:
                    output = await self.chain.ainvoke(payload)

                text = clean_generated_text(output)
                if not text:
                    raise ValueError("Model returned empty text")
                return index, text, None
            except Exception as exc:
                if attempt >= self.config.max_retries:
                    return index, None, f"{exc.__class__.__name__}: {exc}"

                sleep_seconds = self.config.retry_backoff_seconds * attempt + random.uniform(0.0, 0.5)
                await asyncio.sleep(sleep_seconds)

        return index, None, "Unknown generation failure"

    def sync_merged_json(
        self,
        base_data: list[dict[str, Any]],
        generated_map: dict[int, str],
    ) -> None:
        merged = []
        for idx, item in enumerate(base_data):
            updated_item = dict(item)
            if idx in generated_map:
                updated_item["annotations"] = generated_map[idx]
            merged.append(updated_item)

        write_json_atomic(self.config.output_json_path, merged)

    async def run(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        data = load_json(self.config.input_json_path)
        total = len(data)

        start_index = max(int(self.config.start_index), 0)
        end_index = total if self.config.end_index is None else min(int(self.config.end_index), total)
        if end_index < start_index:
            raise ValueError(f"end_index ({end_index}) must be >= start_index ({start_index})")

        generated_map = load_progress(self.config.progress_jsonl_path)
        pending_indices = [idx for idx in range(start_index, end_index) if idx not in generated_map]

        print(f"Input JSON: {self.config.input_json_path}")
        print(f"Output JSON: {self.config.output_json_path}")
        print(f"Progress JSONL: {self.config.progress_jsonl_path}")
        print(f"Error JSONL: {self.config.error_jsonl_path}")
        print(f"Model: {self.config.model_name}")
        print(f"Range: [{start_index}, {end_index}) / total={total}")
        print(f"Already completed in range: {end_index - start_index - len(pending_indices)}")
        print(f"Pending in range: {len(pending_indices)}")

        if self.config.sync_merged_json_on_flush and generated_map:
            self.sync_merged_json(data, generated_map)

        if not pending_indices:
            print("Nothing to generate. Resume state is already complete for this range.")
            return

        semaphore = asyncio.Semaphore(max(self.config.max_concurrency, 1))
        success_count = 0
        failure_count = 0

        for chunk_start in range(0, len(pending_indices), self.config.flush_every):
            chunk_indices = pending_indices[chunk_start : chunk_start + self.config.flush_every]
            tasks = [self.generate_one(index=idx, item=data[idx], semaphore=semaphore) for idx in chunk_indices]
            chunk_results = await asyncio.gather(*tasks)

            progress_records: list[dict[str, Any]] = []
            error_records: list[dict[str, Any]] = []

            for idx, text, error in chunk_results:
                metadata = data[idx].get("metadata", {})
                if text is not None:
                    generated_map[idx] = text
                    progress_records.append(
                        {
                            "index": idx,
                            "text": text,
                            "element": metadata.get("element", ""),
                            "group": metadata.get("group", ""),
                            "created_at": utc_now_iso(),
                        }
                    )
                    success_count += 1
                else:
                    error_records.append(
                        {
                            "index": idx,
                            "element": metadata.get("element", ""),
                            "group": metadata.get("group", ""),
                            "error": error,
                            "created_at": utc_now_iso(),
                        }
                    )
                    failure_count += 1

            append_jsonl(self.config.progress_jsonl_path, progress_records)
            append_jsonl(self.config.error_jsonl_path, error_records)

            if self.config.sync_merged_json_on_flush:
                self.sync_merged_json(data, generated_map)

            processed_so_far = min(chunk_start + len(chunk_indices), len(pending_indices))
            print(
                f"[flush] processed={processed_so_far}/{len(pending_indices)}, "
                f"success={success_count}, failure={failure_count}, "
                f"saved={len(progress_records)}"
            )

            await asyncio.sleep(0.1)

        self.sync_merged_json(data, generated_map)
        print("Done.")
        print(f"Final success count in this run: {success_count}")
        print(f"Final failure count in this run: {failure_count}")
        print(f"Merged JSON written to: {self.config.output_json_path}")


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")

    config = RunConfig(
        input_json_path=DATASET_DIR / "fit_dualsg_all.json",
        output_dir=DATASET_DIR / "llm_route2_outputs",
        output_json_filename="fit_dualsg_metadata_aware_llm.json",
        progress_jsonl_filename="fit_dualsg_metadata_aware_llm.progress.jsonl",
        error_jsonl_filename="fit_dualsg_metadata_aware_llm.errors.jsonl",
        start_index=0,
        end_index=None,  # exclusive; e.g. 100 means [0, 100)
        flush_every=10,
        max_concurrency=5,
        max_retries=5,
        retry_backoff_seconds=3.0,
        request_timeout_seconds=120.0,
        temperature=0.2,
        max_tokens=180,
        model_name=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        sync_merged_json_on_flush=True,
    )

    missing = []
    if not config.model_name:
        missing.append("MODEL_NAME")
    if not config.api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not config.base_url:
        missing.append("DEEPSEEK_BASE_URL")
    if missing:
        raise EnvironmentError(f"Missing required env values: {', '.join(missing)}")

    asyncio.run(MetadataAwareAnnotationGenerator(config).run())


if __name__ == "__main__":
    main()
