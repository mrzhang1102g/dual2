#!/bin/bash

# GeoStyle num only: 52-26
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  --task_name geo_num \
  --is_training 1 \
  --model_id geostyle_num \
  --model Model_Geo_Num \
  --data Geo_Meta \
  --root_path ./dataset/ \
  --data_path Geo_DualSG/geo_dualsg_all.json \
  --inverse \
  --scale True \
  --seq_len 52 \
  --label_len 0 \
  --pred_len 26 \
  --train_epochs 20 \
  --batch_size 200 \
  --patience 100 \
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/
