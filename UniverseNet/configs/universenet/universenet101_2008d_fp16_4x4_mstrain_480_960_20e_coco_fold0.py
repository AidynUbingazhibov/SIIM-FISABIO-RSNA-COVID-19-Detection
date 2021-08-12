_base_ = [
    '../universenet/models/universenet101_2008d.py',
    '../_base_/datasets/coco_detection_mstrain_480_960.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    backbone=dict(
        init_cfg=None,
        norm_eval=True
        #norm_eval=False
    ),
    bbox_head=dict(
        num_classes=1,
        #norm_cfg=dict(type="SyncBN", requires_grad=True)
        norm_cfg=dict(type="BN", requires_grad=True)
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

data_root = "/home/dcm_folds/"
dataset_type = "CustomDataset"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

evaluation = dict(metric='mAP')

albu_train_transforms = [
    dict(type="RandomBrightnessContrast", p=0.4, brightness_limit=0.25, contrast_limit=0.25),
    dict(type="OneOf",
         transforms=[
             dict(type="CLAHE"),
             dict(type="RandomGamma")
         ],
         p=0.5
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", p=1.0, blur_limit=15),
            dict(type="MotionBlur", blur_limit=15),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75)
        ],
        p=0.5,
    ),

    dict(
        type='Cutout',
        num_holes=10,
        max_h_size=60,
        max_w_size=60,
        p=0.4
    ),

    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        rotate_limit=10,
        border_mode=0,
        # value=img_norm_cfg["mean"][::-1],
        scale_limit=0.2,
        p=0.5
    ),
    dict(type='HorizontalFlip', p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(
        type="Albu",
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_masks="masks", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(type="BboxParams", format="pascal_voc", label_fields=["gt_labels"], min_visibility=0.3),
    ),
    dict(
        type="Resize",
            img_scale=[(416 + 32 * i, 416 + 32 * i) for i in range(25)],
        multiscale_mode="value",
        keep_ratio=True,
    ),

    #dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    #dict(type='Pad', size=(512, 512)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + 'pseudo_data_fold0.pickle',
        img_prefix="/home",
        pipeline=train_pipeline,
        filter_empty_gt=False,
        type=dataset_type,
        classes = ['opacity']),
    val=dict(
        ann_file=data_root + 'data_val_dcm_fold0.pickle',
        img_prefix="/home",
        pipeline=test_pipeline,
        filter_empty_gt=False,
        type=dataset_type,
        classes = ['opacity']),
    test=dict(
        ann_file=data_root + 'data_val_dcm_fold0.pickle',
        img_prefix="/home",
        pipeline=test_pipeline,
        filter_empty_gt = False,
        type=dataset_type,
        classes = ['opacity']))


optimizer = dict(type='SGD', lr=0.01 * 12 / 16, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000, step=[12, 15])
runner = dict(type='EpochBasedRunner', max_epochs=17)
fp16 = dict(loss_scale=512.)

#resume_from="/home/uni101_pseudo/epoch_12.pth"
load_from="/home/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_20201023_epoch_20-3e0d236a.pth"
work_dir="/home/uni101_pseudo/"