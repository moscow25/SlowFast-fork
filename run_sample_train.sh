#!/bin/bash -e
# Example run of inference

model_dir="/home/ubuntu/checkpoints/pretrained/model_zoo"
# Swap to any pre-trained model downloaded (model description & weights)
model_path="SLOWFAST_8x8_R50.pkl"
# Kinetics -- 400 classes
num_classes=400
# PitchType -- 8 classes...
num_classes=8
# Weird stuff with "precise batchnorm stats" -- requires min number of iterations
bn_batches=200 # default
bn_batches=20 # smaller data...
# Checkpoints -- where to save and how often?
check_period=10

cmd="python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  NUM_GPUS 1 TRAIN.ENABLE True TEST.ENABLE False \
  DATA.PATH_TO_DATA_DIR data/edge-100/  \
  TRAIN.CHECKPOINT_FILE_PATH $model_dir/$model_path \
  TRAIN.CHECKPOINT_TYPE caffe2 TRAIN.BATCH_SIZE 4 \
  TRAIN.CHECKPOINT_PERIOD $check_period \
  MODEL.NUM_CLASSES $num_classes \
  BN.NUM_BATCHES_PRECISE $bn_batches"

echo $cmd
$cmd
