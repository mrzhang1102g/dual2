#!/bin/bash

# FIT纯数值流一年预测（不带元数据）

# 激活dualsg2环境
source /data1/miniconda3/bin/activate dualsg2

# 运行训练
python run.py \
  # 任务基本参数
  --task_name fit_num \
  --is_training 1 \
  --model_id fit_oneyear_num \
  --model Model_Fit_Num \
  --data FIT_Meta \
  # 数据参数
  --root_path ./dataset/ \
  --data_path FIT_DualSG/fit_dualsg_all.json \
  --inverse \
  --scale True \
  # 时间序列参数
  --seq_len 48 \
  --label_len 0 \
  --pred_len 24 \
  # 训练参数
  --train_epochs 20 \
  --batch_size 200 \
  # 其他参数
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/