from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from utils import read_xray, resize
import numpy as np
import pickle
import albumentations as A
from matplotlib import pyplot as plt
import torch.nn.functional as F
from matplotlib.patches import Rectangle
import cv2
import torch

class_num = 4
res_im = 680
crop_im = 640
def data_transforms0():
    return {
        'train': albumentations.Compose([
            albumentations.Resize(res_im, res_im),
            albumentations.RandomCrop(crop_im, crop_im),
            albumentations.RandomBrightnessContrast(p = 0.4, brightness_limit=0.25, contrast_limit=0.25),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.OneOf([
                albumentations.CLAHE(),
                albumentations.RandomGamma()
                ], p=1.0
            ),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.2, shift_limit=0.1, p=0.9),
            albumentations.OneOf([
                albumentations.Blur(blur_limit=7, p=1.0),
                albumentations.MotionBlur(),
                albumentations.GaussNoise(),
                albumentations.ImageCompression(quality_lower=75)
            ], p=0.5),

            albumentations.Cutout(num_holes=32, max_h_size=16, max_w_size=16, fill_value=250, p=0.65),
            albumentations.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=["class_labels"])),
        'val': albumentations.Compose([
            albumentations.Resize(res_im, res_im),
            albumentations.CenterCrop(crop_im, crop_im),
            albumentations.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=["class_labels"]))
}


class CustomDataset(Dataset):
    def __init__(self, transform, split, fold, data_root):
        self.transform = transform
        self.data_root = data_root
        self.split = split

        with open(f"/data/zhan/compets/siim_covid/input/dcm_folds/data_{split}_dcm_fold{fold}.pickle", "rb") as f:
            self.bbox_data = pickle.load(f)

        self.classes = []

        for idx in range(len(self.bbox_data)):
            self.classes.append(self.bbox_data[idx]["ann"]["class_labels"])


    def __len__(self):
        return len(self.bbox_data)

    def get_labels(self):
        return self.classes

    def __getitem__(self, idx):

        dcm_img_path = self.bbox_data[idx]["filename"]

        cls = self.classes[idx]
        xray = read_xray(os.path.join(self.data_root, dcm_img_path))
        img = np.stack((xray,) * 3, axis=-1)
        bboxes = self.bbox_data[idx]["ann"]["bboxes"]

        transformed = self.transform(image=img, bboxes = bboxes, class_labels=np.zeros(len(bboxes)))
        img = transformed["image"]

        bboxes = transformed["bboxes"]

        mask = np.zeros_like(img[0])


        #Ellipse mask
        #for bbox in bboxes:
        #   x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        #   mask = cv2.ellipse(mask, (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0.0,
        #                      0.0, 360.0, (1, 1, 1), -1)


        # Octagon mask
        for bbox in bboxes:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            roi_corners = np.array([[(x1, y1 + 1 * (y2 - y1) // 4), (x1, y1 + 3 * (y2 - y1) // 4),
                                     (x1 + 1 * (x2 - x1) // 4, y2), (x1 + 3 * (x2 - x1) // 4, y2),
                                     (x2, y1 + 3 * (y2 - y1) // 4), (x2, y1 + 1 * (y2 - y1) // 4),
                                     (x1 + 3 * (x2 - x1) // 4, y1), (x1 + 1 * (x2 - x1) // 4, y1)]], dtype=np.int32)
            mask = cv2.fillPoly(mask, roi_corners, (1, 1, 1))


        #Simple Rectangle
        #for bbox in bboxes:
        #        mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1


        mask0 = cv2.resize(np.array(mask), (crop_im//4, crop_im//4))
        mask01 = cv2.resize(np.array(mask), (crop_im//8, crop_im//8))
        mask1 = cv2.resize(np.array(mask), (crop_im//16, crop_im//16))
        mask2 = cv2.resize(np.array(mask), (crop_im//32, crop_im//32))


        return img, cls, mask0, mask01, mask1, mask2