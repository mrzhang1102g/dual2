#!/bin/bash

# FIT 数值流 + 元数据：半年预测
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  --task_name fit_num_with_meta \
  --is_training 1 \
  --model_id fit_halfyear_num_with_meta \
  --model Model_Fit_Num_With_Meta \
  --data FIT_Meta \
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
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/
