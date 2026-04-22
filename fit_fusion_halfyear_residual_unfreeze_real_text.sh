#!/bin/bash

# FIT fusion reference run: residual + load numerical ckpt + real text captions
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  --task_name fit_fusion \
  --is_training 1 \
  --model_id fit_fusion_halfyear_residual_unfreeze_real_text \
  --model Model_Fit_Fusion \
  --data FIT_Fusion \
  --root_path ./dataset/ \
  --data_path FIT_DualSG/fit_dualsg_all.json \
  --inverse \
  --scale True \
  --fit_scaler_mode train_only \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 12 \
  --train_epochs 20 \
  --batch_size 200 \
  --patience 100 \
  --num_model_path ./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth \
  --caption_emb_path ./dataset/FIT_DualSG/pt/fit_dualsg_all.pt \
  --text_mode residual \
  --fusion_optimizer_mode split \
  --adjust 0 \
  --learning_rate 0.001 \
  --lr_num 0.0001 \
  --lr_text 0.0005 \
  --num_feat_dim 64 \
  --fusion_hidden 128 \
  --fusion_dropout 0.1 \
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/
