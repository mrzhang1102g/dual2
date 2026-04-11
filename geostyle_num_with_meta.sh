#!/bin/bash

# GeoStyle纯数值流预测（带元数据）

# 激活dualsg2环境
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  # 任务基本参数
  --task_name geo_num_with_meta \
  --is_training 1 \
  --model_id geostyle_num_with_meta \
  --model Model_Geo_Num_With_Meta \
  --data Geo_Meta \
  # 数据参数
  --root_path ./dataset/ \
  --data_path Geo_DualSG/geo_dualsg_all.json \
  --inverse \
  --scale True \
  # 时间序列参数
  --seq_len 52 \
  --label_len 0 \
  --pred_len 26 \
  # 训练参数
  --train_epochs 20 \
  --batch_size 32 \
  # GeoStyle特有参数
  --use_element \
  --use_group \
  # 其他参数
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/