import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import math
import argparse

text_encoder_name = "/root/zj/project/DualSG_refined/weights/gpt2"

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
args = parser.parse_args()

DATA_PATH = args.data_path
SAVE_PATH = args.save_path

print(f"Loading model:{text_encoder_name}")

# 指定使用GPU 2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
model = AutoModel.from_pretrained(text_encoder_name).cuda().eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------- 总进度（阶段级）--------
overall = tqdm(total=3, desc="总进度", unit="stage")

# -------- 读取 JSON --------
print(f"Loading JSON:{DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total samples:", len(data))

# -------- 收集 caption 文本（带进度）--------
texts = []
for item in tqdm(data, desc="收集caption", unit="sample"):
    ann = item.get("annotations", "")

    # 你的新annotations是schema JSON字符串，这里优先解析取caption
    if isinstance(ann, str):
        ann_str = ann.strip()
        if ann_str.startswith("{") and ann_str.endswith("}"):
            try:
                obj = json.loads(ann_str)
                # 优先取caption，如果没有则取其他字段
                cap = obj.get("caption", "")
                if not cap:
                    # 对于event_centric，取major_turning_points的描述
                    if "major_turning_points" in obj:
                        turning_points = obj["major_turning_points"]
                        cap = " ".join([f"at time {tp.get('t', '')}, {tp.get('type', '')} with prominence {tp.get('prominence', '')}" for tp in turning_points])
                    # 对于global_structural，取overall_trend和recent_regime
                    elif "overall_trend" in obj:
                        cap = f"overall trend: {obj.get('overall_trend', '')}, recent regime: {obj.get('recent_regime', '')}"
                    # 对于dynamic_behavior，取volatility和volatile_segments
                    elif "volatility" in obj:
                        cap = f"volatility: {obj.get('volatility', '')}"
                        if "volatile_segments" in obj:
                            segments = obj["volatile_segments"]
                            seg_desc = " ".join([f"segment {s.get('start_t', '')}-{s.get('end_t', '')}: {s.get('note', '')}" for s in segments])
                            cap += f", {seg_desc}"
                texts.append(cap if isinstance(cap, str) else "")
                continue
            except Exception:
                pass
        # 兜底：当普通字符串处理
        texts.append(ann_str)

    elif isinstance(ann, list):
        texts.append(" ".join([str(x) for x in ann]))
    else:
        texts.append(str(ann))

overall.update(1)

# -------- GPU 加速：批量计算（带进度）--------
batch_size = 128  # 可调，减小batch_size以解决内存不足问题
all_embs = []

num_batches = math.ceil(len(texts) / batch_size)
print(f"Encoding captions in batches... batch_size={batch_size}, num_batches={num_batches}")

for i in tqdm(range(0, len(texts), batch_size), desc="编码batch", unit="batch"):
    batch_text = texts[i : i + batch_size]

    tokens = tokenizer(
        batch_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**tokens)
        pooled = outputs.last_hidden_state.mean(dim=1)  # [B, D]

    all_embs.append(pooled.cpu())

overall.update(1)

# -------- 合并 + 保存（阶段进度）--------
all_embs = torch.cat(all_embs, dim=0)
print("Final embedding shape:", all_embs.shape)

print(f"Saving embeddings to:{SAVE_PATH}")
torch.save(all_embs, SAVE_PATH)
print("Saved to:", SAVE_PATH)

overall.update(1)
overall.close()
