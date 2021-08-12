## SIIM-FISABIO-RSNA-COVID-19-Detection 7th place solution

# Validation:

We used iterative-stratification with 5 folds (https://github.com/trent-b/iterative-stratification) stratified by study level classes and number of boxes.

# Study level:

We used efficientnet-b7, v2s, v2m, and v2l with aux branches after 3 different blocks. The models were trained on 3 folds and on different image resolutions (512, 640, 768) to produce 14 classifiers.

We used simple averaging for ensembling the models. LB mAP for study level was ~41.5-41.6

# Image level:

We used mmdetection library to train detectoRS50, universeNet50, and universeNet101. detectoRS50 and universeNet50 were trained on one fold, and universeNet101 was trained on each fold + pseudo labels for public data using the universeNet50 model.

WBF did not work for us, so we decided to use NMW for ensembling from https://github.com/ZFTurbo/Weighted-Boxes-Fusion.

TTA: HorizontalFlip for detectoRS and multi-scale TTA for all universeNet models on [(640, 640), (800, 800)].

Binary classifiers were trained in the same manner as study level models, 3 fold ensemble was used.

# Augmentations:

Our augmentations include HorizontalFlip, RandomCrop (for study level), ShiftScaleRotate, CLAHE, RandomGamma, Cutout from albumentations library (https://albumentations.ai/ ).
