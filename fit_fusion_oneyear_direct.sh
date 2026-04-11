#!/bin/bash

# FIT融合模型一年预测（direct模式）

# 激活dualsg2环境
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  # 任务基本参数
  --task_name fit_fusion \
  --is_training 1 \
  --model_id fit_fusion_oneyear_direct \
  --model Model_Fit_Fusion \
  --data FIT_Fusion \
  # 数据参数
  --root_path ./dataset/ \
  --data_path FIT_DualSG/fit_dualsg_all.json \
  --inverse \
  --scale True \
  # 时间序列参数
  --seq_len 48 \
  --label_len 0 \
  --pred_len 24 \
  # 融合模型参数
  --caption_emb_path ./dataset/FIT_DualSG/fit_caption_emb_all.pt \
  --text_mode direct \
  --llm_dim 768 \
  --num_feat_dim 26 \
  --patch_adaptive 1 \
  # 其他参数
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/