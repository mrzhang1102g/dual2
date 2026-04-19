#!/bin/bash

# FIT half-year semantic supervision v1
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  --task_name fit_num_with_meta_semantic \
  --is_training 1 \
  --model_id fit_halfyear_num_with_meta_semantic_v1 \
  --model Model_Fit_Num_With_Meta_Semantic \
  --data FIT_Meta_Semantic \
  --root_path ./dataset/ \
  --data_path FIT_DualSG/fit_dualsg_structured.json \
  --inverse \
  --scale True \
  --fit_scaler_mode train_only \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 12 \
  --train_epochs 20 \
  --batch_size 200 \
  --patience 100 \
  --pretrained_num_model_path ./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth \
  --semantic_text_field annotations \
  --semantic_hidden 64 \
  --semantic_dropout 0.1 \
  --semantic_loss_weight 0.2 \
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/
