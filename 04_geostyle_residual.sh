#!/bin/bash

# GeoStyle: residual fusion
source /data1/miniconda3/bin/activate dualsg2

NUM_CKPT=./model_checkpoints/REPLACE_WITH_02_GEOSTYLE_NUM_WITH_META/checkpoint.pth

python run.py \
  --task_name geo_fusion \
  --is_training 1 \
  --model_id geostyle_residual \
  --model Model_Geo_Fusion \
  --data Geo_Fusion \
  --root_path ./dataset/ \
  --data_path Geo_DualSG/geo_dualsg_all.json \
  --inverse \
  --scale True \
  --seq_len 52 \
  --label_len 0 \
  --pred_len 26 \
  --train_epochs 100 \
  --batch_size 200 \
  --patience 100 \
  --num_model_path "$NUM_CKPT" \
  --caption_emb_path ./dataset/Geo_DualSG/pt/geo_dualsg_all.pt \
  --text_mode residual \
  --fusion_optimizer_mode split \
  --learning_rate 0.001 \
  --lr_num 0.0001 \
  --lr_text 0.0005 \
  --num_feat_dim 64 \
  --fusion_hidden 128 \
  --patch_adaptive 1 \
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/
