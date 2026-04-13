#!/bin/bash

# FIT fusion 官方 recipe：joint_direct（一年）
source /data1/miniconda3/bin/activate dualsg2

DATA_PATH=${DATA_PATH:-FIT_DualSG/fit_dualsg_all.json}
CAPTION_EMB_PATH=${CAPTION_EMB_PATH:-./dataset/FIT_DualSG/fit_caption_emb_all.pt}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-200}
PATIENCE=${PATIENCE:-30}
ADJUST=${ADJUST:-0}
LEARNING_RATE=${LEARNING_RATE:-0.001}
LR_NUM=${LR_NUM:-0.0001}
LR_TEXT=${LR_TEXT:-0.0005}
NUM_FEAT_DIM=${NUM_FEAT_DIM:-64}
FUSION_HIDDEN=${FUSION_HIDDEN:-128}
FUSION_DROPOUT=${FUSION_DROPOUT:-0.1}

python run.py \
  --task_name fit_fusion \
  --is_training 1 \
  --model_id fit_fusion_oneyear_joint_direct \
  --model Model_Fit_Fusion \
  --data FIT_Fusion \
  --root_path ./dataset/ \
  --data_path "$DATA_PATH" \
  --inverse \
  --scale True \
  --fit_scaler_mode train_only \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 24 \
  --train_epochs "$TRAIN_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --patience "$PATIENCE" \
  --num_model_path "" \
  --caption_emb_path "$CAPTION_EMB_PATH" \
  --text_mode direct \
  --fusion_optimizer_mode split \
  --adjust "$ADJUST" \
  --learning_rate "$LEARNING_RATE" \
  --lr_num "$LR_NUM" \
  --lr_text "$LR_TEXT" \
  --num_feat_dim "$NUM_FEAT_DIM" \
  --fusion_hidden "$FUSION_HIDDEN" \
  --fusion_dropout "$FUSION_DROPOUT" \
  --visualize False \
  --output_dir ./model_outputs/ \
  --checkpoint_dir ./model_checkpoints/
