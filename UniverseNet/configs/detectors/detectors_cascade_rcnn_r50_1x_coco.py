_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,

    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))

        ]
    ),
    backbone=dict(
        type='DetectoRS_ResNet',
        plugins=[
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='1111',
                    kv_stride=2),
                stages=(False, False, True, True),
                position='after_conv2')
        ],
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            #pretrained='torchvision://resnet50',
            pretrained=None,
            style='pytorch')),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type="nms", iou_threshold=0.5)
        )
    )
)

data_root = "/home/"
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
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'dcm_folds/pseudo_data_fold4.pickle',
        img_prefix="/home/",
        pipeline=train_pipeline,
        filter_empty_gt=False,
        type=dataset_type,
        classes = ['opacity']),
    val=dict(
        ann_file=data_root + 'dcm_folds/data_val_dcm_fold4.pickle',
        img_prefix="/home/",
        pipeline=test_pipeline,
        filter_empty_gt=False,
        type=dataset_type,
        classes = ['opacity']),
    test=dict(
        ann_file=data_root + 'dcm_folds/data_val_dcm_fold4.pickle',
        img_prefix="/home/",
        pipeline=test_pipeline,
        filter_empty_gt = False,
        type=dataset_type,
        classes = ['opacity']))

#optimizer = dict(lr=0.02 / 8)

#lr_config = dict(
#    _delete_=True,
#    policy='CosineAnnealing',
#    warmup='linear',
#    warmup_iters=1000,
#    warmup_ratio=1.0 / 10,
#    min_lr_ratio=1e-5)
optimizer = dict(type='SGD', lr=0.02 * 4 * 1 / 16, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000, step=[12, 15])
runner = dict(type='EpochBasedRunner', max_epochs=17)

#load_from="/home/models/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth"
#resume_from="/mmdetection/work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_15.pth"
resume_from="/home/det50_pseudo/epoch_14.pth"
work_dir="/home/det50_pseudo"
#resume_from="/UniverseNet/work_dirs/universenet50_gfl_fp16_4x4_mstrain_480_960_1x_coco/epoch_12.pth"