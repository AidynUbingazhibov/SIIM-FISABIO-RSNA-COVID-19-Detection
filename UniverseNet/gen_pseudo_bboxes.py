import sys

sys.path.insert(0, "/home/UniverseNet")

from ensemble_boxes import *
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
import pickle
import pandas as pd
import os
import json
import numpy as np
import glob
from scipy.stats import rankdata
from random import randrange
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
from tqdm import tqdm

import torch
import os
import time
import random
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
import glob

from tqdm import tqdm
import torchvision
'''
def gen_test_annotation(annotation_path):
    img_paths = glob.glob("/home/test_dcm/*")
    test_anno_list = []
    images = []
    i = 0
    flag = 0

    for img_path in tqdm(img_paths):
        if True:
            if True:
                img_info = {}
                img_info['filename'] = img_path
                images.append(img_path)
                # img_size = pydicom.read_file(img).pixel_array.shape
                img_shape = plt.imread(img_path).shape
                img_info['width'] = img_shape[1]
                img_info['height'] = img_shape[0]
                test_anno_list.append(img_info)
            i += 1

    with open(annotation_path, 'wb') as f:
        pickle.dump(test_anno_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    return images
'''

def gen_test_annotation(annotation_path):
    study_paths = glob.glob("/home/test_dcm/*")
    test_anno_list = []
    duplicate_idxs = []
    images = []
    i = 0
    flag = 0
    image2study = {}

    for study_path in tqdm(study_paths):
        imgs = glob.glob(study_path + "/*/*")

        if len(imgs) > 1:
            flag = 1
            dups = []

        for img in imgs:
            if flag == 1:
                dups.append(i)

            if img.endswith('dcm'):
                img_info = {}
                img_info['filename'] = img
                images.append(img)
                img_info['studyname'] = study_path.split("/")[-1]
                img_size = pydicom.read_file(img).pixel_array.shape
                img_info['width'] = img_size[1]
                img_info['height'] = img_size[0]
                test_anno_list.append(img_info)

            i += 1
        if flag == 1:
            duplicate_idxs.append(dups)
            flag = 0
    with open(annotation_path, 'wb') as f:
        pickle.dump(test_anno_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    return images, duplicate_idxs

#images = gen_test_annotation("test_annotation_midrc.pickle")
images = gen_test_annotation("detection_test.pickle")
print(len(images))

#config_file_uni = "/UniverseNet/configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco.py"
config_file_uni = "/UniverseNet/configs/universenet/universenet50_gfl_fp16_4x4_mstrain_480_960_1x_coco.py"
cfg_uni = Config.fromfile(config_file_uni)
cfg_uni.data.test.test_mode = True
distributed = False

dataset = build_dataset(cfg_uni.data.test)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=distributed,
    shuffle=False)

#DIR_WEIGHTS = '/home/uni101_pseudo'
#WEIGHTS_FILE = f'{DIR_WEIGHTS}/map_55.pth'
DIR_WEIGHTS = "/home/uni"
WEIGHTS_FILE = f"{DIR_WEIGHTS}/universenet50_gfl_54_6.pth"
model = build_detector(cfg_uni.model, train_cfg=None, test_cfg=None)
checkpoint = load_checkpoint(model, WEIGHTS_FILE, map_location='cuda')

model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
outputs_uni = single_gpu_test(model, data_loader, False, None, 0.5)

np.save("outputs_uni50_fold4.npy", outputs_uni)