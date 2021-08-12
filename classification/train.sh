#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0,2,3,4 python train_normal.py --scheduler plateau --fold 4 --aux 5678 --v2_size m --batch_size 32
CUDA_VISIBLE_DEVICES=0,2,3,4 python train_normal.py --scheduler plateau --fold 2 --aux 5678 --v2_size m --batch_size 32

