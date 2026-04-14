#!/usr/bin/env python3
"""
Generate metadata-aware English annotations for FIT_DualSG with LangChain.

Route-2 input:
- historical series
- metadata (city / gender / age group / element)

Design choices:
- plain-text output only, no structured parsing
- range-scoped JSONL progress file for resume
- final JSON contains only the selected index slice
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

AGE_LABEL_MAP = {
    "age_lt_18": "under 18",
    "age_18_25": "18 to 25",
    "age_25_40": "25 to 40",
    "age_gt_40": "over 40",
}

SYSTEM_PROMPT_EN = """You write concise English annotations for weekly fashion trend windows.

Use the metadata as domain context, but base every claim on the provided historical series only.
Do not predict future target values.
Prefer wording that naturally combines the fashion element, location, and consumer group with the observed pattern.
Do not call the 48-week window a calendar year, a season, a month, or a specific date range unless such information is explicitly provided.
Vary phrasing modestly across samples and avoid repetitive stock openings when possible.
Do not mention model training, prompts, hidden features, or uncertainty boilerplate.
Return exactly one short paragraph in natural English.
No bullet points, no JSON, no markdown, no title.
"""

# 中文翻译，仅供人工对照，不参与实际调用。
SYSTEM_PROMPT_ZH = """你要为按周组织的时尚趋势窗口写简洁的英文描述。

可以把元数据当作领域上下文，但所有判断都必须基于给定的历史序列本身。
不要预测未来 target。
尽量把时尚元素、城市和消费者群体自然地融合到描述里。
不要把这个 48 周窗口说成某个自然年、某个季节、某个月份，或具体日期范围，除非输入里明确给出。
措辞可以有适度变化，尽量避免每条都用同一种模板化开头。
不要提模型训练、提示词、隐藏特征，也不要写空泛免责声明。
只返回一段简短自然的英文段落。
不要项目符号，不要 JSON，不要 Markdown，不要标题。
"""

USER_PROMPT_EN = """Task
Write one short English paragraph for this 48-week historical fashion trend window.

Metadata
- Fashion element: {element_phrase}
- Element value: {element_value}
- Element category: {element_category}
- City: {city}
- Gender: {gender}
- Age group: {age_group}

Window context
- Frequency: weekly
- Historical window length: {series_length} weeks
- When useful, refer to the timing of movements as early, middle, or late in the window, and mention recent weeks separately.

Series values
[{series_values}]

Key cues
- Overall trend: {overall_trend}
- Recent trend: {recent_trend}
- Volatility: {volatility_tag}
- Value range: {min_value} to {max_value}
- Peak timing: {peak_position} (value {peak_value})
- Trough timing: {trough_position} (value {trough_value})

Requirements
1. Use the metadata as meaningful context, not as filler.
2. Describe the observed historical pattern only.
3. Mention the overall pattern, volatility, and recent behavior.
4. Keep it specific and fluent, not generic.
5. Prefer 2 to 4 sentences and roughly 40 to 90 words.
6. Use the element, city, gender, and age group naturally in the paragraph rather than listing them mechanically.
7. Return only the paragraph text.
"""

# 中文翻译，仅供人工对照，不参与实际调用。
USER_PROMPT_ZH = """任务
请为这个 48 周历史时尚趋势窗口写一段简短英文描述。

元数据
- Fashion element: {element_phrase}
- Element value: {element_value}
- Element category: {element_category}
- City: {city}
- Gender: {gender}
- Age group: {age_group}

窗口上下文
- 频率: 周
- 历史窗口长度: {series_length} 周
- 如有需要，可以用 early / middle / late 来描述变化发生在窗口中的位置，并单独提 recent weeks。

序列数值
[{series_values}]

关键提示
- 整体趋势: {overall_trend}
- 近期趋势: {recent_trend}
- 波动程度: {volatility_tag}
- 数值范围: {min_value} 到 {max_value}
- 峰值时段: {peak_position} (value {peak_value})
- 谷值时段: {trough_position} (value {trough_value})

要求
1. 元数据要真正作为上下文，而不是填充废话。
2. 只描述已观察到的历史模式。
3. 需要提到整体模式、波动情况和近期行为。
4. 表达要具体、流畅，不要泛泛而谈。
5. 优先写成 2 到 4 句，约 40 到 90 个英文词。
6. 要把元素、城市、性别、年龄段自然融合到段落里，不要机械罗列。
7. 只返回段落文本。
"""


@dataclass
class RunConfig:
    input_json_path: Path
    output_dir: Path
    output_prefix: str
    total_chunks: int = 8
    chunk_id: int = 0
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
    sync_slice_json_on_flush: bool = True


@dataclass(frozen=True)
class ResolvedPaths:
    range_tag: str
    chunk_tag: str
    output_json_path: Path
    progress_jsonl_path: Path
    error_jsonl_path: Path


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


def prettify_text_label(value: str) -> str:
    text = str(value or "unknown").strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_element(element: str) -> dict[str, str]:
    raw = str(element or "unknown").strip()

    if ":" in raw:
        category, value = raw.split(":", 1)
    elif "__" in raw:
        category, value = raw.split("__", 1)
    else:
        category, value = "attribute", raw

    category_label = prettify_text_label(category)
    value_label = prettify_text_label(value)
    element_phrase = f"{value_label} ({category_label})" if category_label else value_label

    return {
        "element_raw": raw,
        "element_category": category_label or "attribute",
        "element_value": value_label or "unknown",
        "element_phrase": element_phrase or "unknown",
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def build_range_tag(start_index: int, end_index: int) -> str:
    return f"{start_index:06d}_{end_index:06d}"


def resolve_chunk_range(total: int, total_chunks: int, chunk_id: int) -> tuple[int, int]:
    if total_chunks <= 0:
        raise ValueError(f"total_chunks must be positive, got {total_chunks}")
    if chunk_id < 0 or chunk_id >= total_chunks:
        raise ValueError(f"chunk_id must be in [0, {total_chunks}), got {chunk_id}")

    start_index = (total * chunk_id) // total_chunks
    end_index = (total * (chunk_id + 1)) // total_chunks
    return start_index, end_index


def build_resolved_paths(config: RunConfig, start_index: int, end_index: int) -> ResolvedPaths:
    range_tag = build_range_tag(start_index, end_index)
    chunk_tag = f"chunk{config.chunk_id:02d}of{config.total_chunks:02d}"
    prefix = f"{config.output_prefix}_{chunk_tag}_{range_tag}"
    return ResolvedPaths(
        range_tag=range_tag,
        chunk_tag=chunk_tag,
        output_json_path=config.output_dir / f"{prefix}.json",
        progress_jsonl_path=config.output_dir / f"{prefix}.progress.jsonl",
        error_jsonl_path=config.output_dir / f"{prefix}.errors.jsonl",
    )


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

    overall_change = float(arr[-1] - arr[0])
    mean_abs_diff = float(np.mean(np.abs(diffs))) if diffs.size > 0 else 0.0
    recent_mean = float(arr[-recent_window:].mean())
    prev_mean = float(prev_window.mean()) if prev_window.size > 0 else float(arr[0])
    recent_change = recent_mean - prev_mean

    peak_idx = int(np.argmax(arr))
    trough_idx = int(np.argmin(arr))

    return {
        "series_length": int(arr.size),
        "series_values": ", ".join(format_float(v) for v in arr.tolist()),
        "min_value": format_float(float(arr.min())),
        "max_value": format_float(float(arr.max())),
        "overall_trend": classify_trend(overall_change, span),
        "recent_trend": classify_trend(recent_change, span),
        "volatility_tag": classify_volatility(float(arr.std(ddof=0)), mean_abs_diff, span),
        "peak_position": position_label(peak_idx, int(arr.size)),
        "peak_value": format_float(float(arr[peak_idx])),
        "trough_position": position_label(trough_idx, int(arr.size)),
        "trough_value": format_float(float(arr[trough_idx])),
    }


def build_prompt_payload(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    group_info = parse_group(str(metadata.get("group", "")))
    element_info = parse_element(str(metadata.get("element", "unknown")))
    summary = summarize_series(item["series"])

    payload = {
        "element_phrase": element_info["element_phrase"],
        "element_value": element_info["element_value"],
        "element_category": element_info["element_category"],
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

    replacements = {
        r"\bover the past year\b": "over the window",
        r"\bbegan the year\b": "began the window",
        r"\bthroughout the year\b": "throughout the window",
        r"\bduring the year\b": "during the window",
        r"\bwithin the year\b": "within the window",
    }
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

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


def load_errors(path: Path) -> dict[int, str]:
    errors: dict[int, str] = {}
    if not path.exists():
        return errors

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = int(record["index"])
            errors[index] = str(record.get("error", "Unknown error"))

    return errors


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


class MetadataAwareAnnotationGenerator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.paths: ResolvedPaths | None = None
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

    def write_slice_json(
        self,
        selected_items: list[dict[str, Any]],
        generated_map: dict[int, str],
        error_map: dict[int, str],
        start_index: int,
    ) -> None:
        if self.paths is None:
            raise RuntimeError("Resolved output paths are not initialized")

        output_items = []
        for offset, item in enumerate(selected_items):
            global_idx = start_index + offset
            updated_item = dict(item)
            if global_idx in generated_map:
                updated_item["annotations"] = generated_map[global_idx]
            elif global_idx in error_map:
                updated_item["annotations"] = {
                    "error": error_map[global_idx],
                    "index": global_idx,
                    "status": "failed_after_retries",
                }
            output_items.append(updated_item)

        write_json_atomic(self.paths.output_json_path, output_items)

    async def run(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        data = load_json(self.config.input_json_path)
        total = len(data)

        start_index, end_index = resolve_chunk_range(
            total=total,
            total_chunks=int(self.config.total_chunks),
            chunk_id=int(self.config.chunk_id),
        )

        self.paths = build_resolved_paths(self.config, start_index, end_index)
        selected_items = data[start_index:end_index]

        generated_map = load_progress(self.paths.progress_jsonl_path)
        generated_map = {idx: text for idx, text in generated_map.items() if start_index <= idx < end_index}
        error_map = load_errors(self.paths.error_jsonl_path)
        error_map = {idx: err for idx, err in error_map.items() if start_index <= idx < end_index}
        pending_indices = [idx for idx in range(start_index, end_index) if idx not in generated_map and idx not in error_map]

        print(f"Input JSON: {self.config.input_json_path}")
        print(f"Output JSON: {self.paths.output_json_path}")
        print(f"Progress JSONL: {self.paths.progress_jsonl_path}")
        print(f"Error JSONL: {self.paths.error_jsonl_path}")
        print(f"Model: {self.config.model_name}")
        print(f"Chunk: {self.config.chunk_id}/{self.config.total_chunks - 1} ({self.paths.chunk_tag})")
        print(f"Range: [{start_index}, {end_index}) / total={total}")
        print(f"Slice size: {len(selected_items)}")
        print(f"Already completed in range: {end_index - start_index - len(pending_indices)}")
        print(f"Pending in range: {len(pending_indices)}")

        if self.config.sync_slice_json_on_flush:
            self.write_slice_json(selected_items, generated_map, error_map, start_index)

        if not pending_indices:
            print("Nothing to generate. Resume state is already complete for this range.")
            return

        semaphore = asyncio.Semaphore(max(self.config.max_concurrency, 1))
        success_count = 0
        failure_count = 0
        already_completed = end_index - start_index - len(pending_indices)

        with tqdm(
            total=end_index - start_index,
            initial=already_completed,
            desc=f"Generate {self.paths.range_tag}",
            unit="sample",
            dynamic_ncols=True,
        ) as progress_bar:
            progress_bar.set_postfix(success=0, failure=0)

            for chunk_start in range(0, len(pending_indices), self.config.flush_every):
                chunk_indices = pending_indices[chunk_start : chunk_start + self.config.flush_every]
                tasks = [
                    asyncio.create_task(self.generate_one(index=idx, item=data[idx], semaphore=semaphore))
                    for idx in chunk_indices
                ]
                chunk_results: list[tuple[int, str | None, str | None]] = []

                for finished_task in asyncio.as_completed(tasks):
                    result = await finished_task
                    chunk_results.append(result)
                    progress_bar.update(1)

                progress_records: list[dict[str, Any]] = []
                error_records: list[dict[str, Any]] = []

                for idx, text, error in chunk_results:
                    metadata = data[idx].get("metadata", {})
                    if text is not None:
                        generated_map[idx] = text
                        error_map.pop(idx, None)
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
                                "text": {
                                    "error": error,
                                    "index": idx,
                                    "status": "failed_after_retries",
                                },
                                "element": metadata.get("element", ""),
                                "group": metadata.get("group", ""),
                                "error": error,
                                "created_at": utc_now_iso(),
                            }
                        )
                        error_map[idx] = error
                        failure_count += 1

                append_jsonl(self.paths.progress_jsonl_path, progress_records)
                append_jsonl(self.paths.error_jsonl_path, error_records)

                if self.config.sync_slice_json_on_flush:
                    self.write_slice_json(selected_items, generated_map, error_map, start_index)

                processed_so_far = min(chunk_start + len(chunk_indices), len(pending_indices))
                progress_bar.set_postfix(success=success_count, failure=failure_count)
                tqdm.write(
                    f"[flush] processed={processed_so_far}/{len(pending_indices)}, "
                    f"success={success_count}, failure={failure_count}, "
                    f"saved={len(progress_records)}"
                )

                await asyncio.sleep(0.1)

        self.write_slice_json(selected_items, generated_map, error_map, start_index)
        print("Done.")
        print(f"Final success count in this run: {success_count}")
        print(f"Final failure count in this run: {failure_count}")
        print(f"Slice JSON written to: {self.paths.output_json_path}")


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")

    config = RunConfig(
        input_json_path=DATASET_DIR / "fit_dualsg_all.json",
        output_dir=DATASET_DIR / "llm_route2_outputs",
        output_prefix="fit_dualsg_metadata_aware_llm",
        total_chunks=8,
        chunk_id=0,
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
        sync_slice_json_on_flush=True,
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
