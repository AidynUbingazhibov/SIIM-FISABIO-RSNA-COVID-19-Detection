# [SIIM-FISABIO-RSNA-COVID-19-Detection](https://www.kaggle.com/c/siim-covid19-detection) 7th place gold medal solution

## Validation:

We used iterative-stratification with 5 folds (https://github.com/trent-b/iterative-stratification) stratified by study level classes and number of boxes.

## Study level:

We used efficientnet-b7, v2s, v2m, and v2l with aux branches after 3 different blocks. The models were trained on 3 folds and on different image resolutions (512, 640, 768) to produce 14 classifiers.

We used simple averaging for ensembling the models. LB mAP for study level was **~62.3-62.4**

## Image level:

We used mmdetection library to train detectoRS50, universeNet50, and universeNet101. detectoRS50 and universeNet50 were trained on one fold, and universeNet101 was trained on each fold + pseudo labels for public data using the universeNet50 model.

WBF did not work for us, so we decided to use NMW for ensembling from https://github.com/ZFTurbo/Weighted-Boxes-Fusion.

TTA: HorizontalFlip for detectoRS and multi-scale TTA for all universeNet models on `[(640, 640), (800, 800)]`.

Binary classifiers were trained in the same manner as study level models, 3 fold ensemble was used.

## Augmentations:

Our augmentations include HorizontalFlip, RandomCrop (for study level), ShiftScaleRotate, CLAHE, RandomGamma, Cutout from albumentations library (https://albumentations.ai/ ).

## How to run detector:

1) Change the paths to your dcm pickle files in the config files for universeNet and detectoRS.
2) The detector can be trained inside the folder UniverseNet as follows:
```bash
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

## How to run classifier:

1) To train classifier run `train_normal.py`. To change the data root, the dcm pickle files, and the fold number use `--data_root`, `--dcm_folds_train`, `--dcm_folds_val`, and `--fold`. The auxilary branches can be changed as well via `--aux`. For example:
```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 python train_normal.py --scheduler plateau --fold 0 --aux 5678 --v2_size m --batch_size 32 --data_root SIIM-FISABIO-RSNA-COVID-19-Detection --dcm_folds_train /dcm_folds/data_train_dcm_fold0.pickle --dcm_folds_val /dcm_folds/data_val_dcm_fold0.pickle
```
