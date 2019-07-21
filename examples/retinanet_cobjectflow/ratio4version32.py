normalize = dict(
    type='SyncBN',
    sync_stats=True,
    group_size=16,
    momentum=0.1,
    frozen=False)

# model settings
model = dict(
    type='RetinaNet',
    pretrained='/mnt/lustre/liguoxuan/weights/mobilenetv2.pth',
    backbone=dict(
        type='MobileNetV2_ImgNet',
        last_feat_channel=160),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 160],
        out_channels=8,
        start_level=0,
        add_extra_convs=True,
        num_outs=5,
        normalize=normalize),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=8,
        stacked_convs=1,
        feat_channels=8,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        anchor_scales=[8]))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.5,
    nms=dict(type='nms', iou_thr=0.3),
    max_per_img=100)
# dataset settings
dataset_type = 'CustomDataset'
#data_root = '/mnt/lustre/liguoxuan/tojson/'
data_root = '/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/data'
img_norm_cfg = dict(
    mean=[103.53, 116.28,123.675], std=[57.375,57.12,58.395], to_rgb=False)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations_v_1/finalversionold.json',
        img_prefix=data_root + '',
        img_scale=(240,320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        with_triple_grey=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/jingtai/VALannotation_ALL.json',
        img_prefix=data_root + '',
        img_scale=(240,320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/jingtai/VALannotation_ALL.json',
        img_prefix=data_root + '',
        img_scale=(240,320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 26
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './workdir/ratio4_32_little_480_320'
load_from = None
resume_from = None 
workflow = [('train', 1)]
