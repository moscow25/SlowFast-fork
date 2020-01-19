#!/bin/bash -e
# Example run of inference

model_dir="/home/ubuntu/checkpoints/pretrained/model_zoo"
# Swap to any pre-trained model downloaded (model description & weights)
model_path="SLOWFAST_8x8_R50.pkl"

cmd="python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  NUM_GPUS 1 TRAIN.ENABLE False \
  DATA.PATH_TO_DATA_DIR data/edge-100/  \
  TEST.CHECKPOINT_FILE_PATH $model_dir/$model_path \
  TEST.CHECKPOINT_TYPE caffe2 TEST.BATCH_SIZE 4"

echo $cmd
$cmd
