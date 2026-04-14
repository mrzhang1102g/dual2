#!/usr/bin/env python3
"""
基于 FIT_DualSG 的 all.json 直接生成 filler-text 对照 PT。

做法：
- 不再读取每条样本自己的真实文本语义
- 所有样本统一使用同一句占位文本
- 仍然严格保持样本数与 all.json 一致

这样生成的 PT：
- 顺序与 all.json 严格一致
- 每条样本都有 embedding
- 但文本侧不携带任何样本级语义信息
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_ENCODER = "/root/zj/project/DualSG_refined/weights/gpt2"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "fit_dualsg_all.json"
DEFAULT_SAVE = Path(__file__).resolve().parents[1] / "pt" / "fit_dualsg_filler_text.pt"
DEFAULT_FILLER = (
    "This is a generic placeholder description for a time series. "
    "It contains no useful forecasting information."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--save_path", type=str, default=str(DEFAULT_SAVE))
    parser.add_argument("--text_encoder_name", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--filler_text", type=str, default=DEFAULT_FILLER)
    parser.add_argument("--cuda_visible_devices", type=str, default="2")
    return parser.parse_args()


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

    print(f"Loading JSON only for sample count/order: {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total samples: {len(data)}")
    texts = [args.filler_text for _ in tqdm(data, desc="Build filler texts", unit="sample")]

    embeddings = encode_texts(
        texts,
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
