#!/bin/bash

# GeoStyle num+meta: 52-26
source /data1/miniconda3/bin/activate dualsg2

python run.py \
  --task_name geo_num_with_meta \
  --is_training 1 \
  --model_id geostyle_num_with_meta \
  --model Model_Geo_Num_With_Meta \
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
  --use_element \
  --use_group \
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/
