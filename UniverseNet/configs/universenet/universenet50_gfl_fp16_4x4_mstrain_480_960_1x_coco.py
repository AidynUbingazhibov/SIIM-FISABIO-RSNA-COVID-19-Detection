_base_ = [
    '../universenet/models/universenet50_gfl.py',
    '../_base_/datasets/coco_detection_mstrain_480_960.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    bbox_head=dict(
        num_classes=1
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

data_root = "/classification/"
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
        ann_file=data_root + 'data_train_dcm.pickle',
        img_prefix="/home/",
        pipeline=train_pipeline,
        filter_empty_gt=False,
        type=dataset_type,
        classes = ['opacity']),
    val=dict(
        ann_file=data_root + 'detection_test.pickle',
        img_prefix="",
        pipeline=test_pipeline,
        filter_empty_gt=False,
        type=dataset_type,
        classes = ['opacity']),
    test=dict(
        ann_file=data_root + 'detection_test.pickle',
        img_prefix="",
        pipeline=test_pipeline,
        filter_empty_gt = False,
        type=dataset_type,
        classes = ['opacity']))


optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)

load_from="/home/uni/universenet50_gfl_54_6.pth"