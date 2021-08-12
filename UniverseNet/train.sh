#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_fold1.py 1
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_fold3.py 1
