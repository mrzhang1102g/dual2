#!/bin/bash

# GeoStyle融合模型预测（residual模式）

# 激活dualsg2环境
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  # 任务基本参数
  --task_name geo_fusion \
  --is_training 1 \
  --model_id geostyle_fusion_residual \
  --model Model_Geo_Fusion \
  --data Geo_Fusion \
  # 数据参数
  --root_path ./dataset/ \
  --data_path Geo_DualSG/geo_dualsg_all.json \
  --inverse \
  --scale True \
  # 时间序列参数
  --seq_len 52 \
  --label_len 0 \
  --pred_len 26 \
  # 融合模型参数
  --caption_emb_path ./dataset/Geo_DualSG/geo_caption_emb_all.pt \
  --text_mode residual \
  --llm_dim 768 \
  --num_feat_dim 52 \
  --patch_adaptive 1 \
  # 其他参数
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/