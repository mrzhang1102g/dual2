#!/usr/bin/env python3
"""
基于 FIT_DualSG 的 all.json 直接生成 random-text 对照 PT。

做法：
- 先按当前项目既有规则抽取每条样本对应的文本描述
- 再在样本之间做一次打乱，但保证尽量不让样本拿回自己的原文本
- 最后用同一套 GPT-2 编码流程得到 embedding，并保存为 PT

这样生成的 PT：
- 样本数与 all.json 严格一致
- 顺序与 all.json 严格一致
- 文本长度/词汇分布接近真实文本
- 但文本和数值序列的语义配对关系被打断
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_ENCODER = "/root/zj/project/DualSG_refined/weights/gpt2"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "fit_dualsg_all.json"
DEFAULT_SAVE = Path(__file__).resolve().parents[1] / "pt" / "fit_dualsg_random_text.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE))
    parser.add_argument("--text_encoder_name", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_visible_devices", type=str, default="2")
    return parser.parse_args()


def extract_caption_text(item):
    ann = item.get("annotations", "")

    if isinstance(ann, str):
        ann_str = ann.strip()
        if ann_str.startswith("{") and ann_str.endswith("}"):
            try:
                obj = json.loads(ann_str)
                cap = obj.get("caption", "")
                if not cap:
                    if "major_turning_points" in obj:
                        turning_points = obj["major_turning_points"]
                        cap = " ".join(
                            [
                                f"at time {tp.get('t', '')}, {tp.get('type', '')} with prominence {tp.get('prominence', '')}"
                                for tp in turning_points
                            ]
                        )
                    elif "overall_trend" in obj:
                        cap = (
                            f"overall trend: {obj.get('overall_trend', '')}, "
                            f"recent regime: {obj.get('recent_regime', '')}"
                        )
                    elif "volatility" in obj:
                        cap = f"volatility: {obj.get('volatility', '')}"
                        if "volatile_segments" in obj:
                            segments = obj["volatile_segments"]
                            seg_desc = " ".join(
                                [
                                    f"segment {s.get('start_t', '')}-{s.get('end_t', '')}: {s.get('note', '')}"
                                    for s in segments
                                ]
                            )
                            cap += f", {seg_desc}"
                return cap if isinstance(cap, str) else ""
            except Exception:
                pass
        return ann_str

    if isinstance(ann, list):
        return " ".join([str(x) for x in ann])

    return str(ann)


def build_derangement(num_items, seed):
    if num_items <= 1:
        raise ValueError("Need at least 2 items to build a random-text derangement.")

    rng = np.random.default_rng(seed)
    for attempt in range(1, 101):
        perm = rng.permutation(num_items)
        if not np.any(perm == np.arange(num_items)):
            print(f"Built derangement after {attempt} attempt(s).")
            return perm

    raise RuntimeError("Failed to build a derangement after 100 attempts.")


def encode_texts(texts, tokenizer, model, device, batch_size, max_length):
    all_embs = []
    num_batches = math.ceil(len(texts) / batch_size)
    print(f"Encoding texts in batches... batch_size={batch_size}, num_batches={num_batches}")

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batch", unit="batch"):
        batch_text = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            pooled = outputs.last_hidden_state.mean(dim=1)

        all_embs.append(pooled.cpu())

    return torch.cat(all_embs, dim=0)


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.text_encoder_name}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
    model = AutoModel.from_pretrained(args.text_encoder_name).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading JSON: {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total samples: {len(data)}")

    texts = [extract_caption_text(item) for item in tqdm(data, desc="Collect caption", unit="sample")]
    permutation = build_derangement(len(texts), seed=args.seed)
    shuffled_texts = [texts[index] for index in permutation]

    embeddings = encode_texts(
        shuffled_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Final embedding shape: {tuple(embeddings.shape)}")
    print(f"Saving embeddings to: {save_path}")
    torch.save(embeddings, save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
