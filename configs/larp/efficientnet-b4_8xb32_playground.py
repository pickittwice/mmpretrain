_base_ = [
    '../_base_/models/efficientnet_b4.py',
    '../_base_/datasets/larp_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256, keep_ratio=True),
    dict(type='RandomCrop', crop_size=256, pad_if_needed=True),
    dict(type='Rotate', 
         magnitude_level=3,
         magnitude_range=(0, 30), 
         magnitude_std='inf',
         random_negative_prob=0.5,
         pad_val=0),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.0)
        ],
        prob=0.8),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256, keep_ratio=True),
    dict(type='RandomCrop', crop_size=256, pad_if_needed=True),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
    batch_size=16
    )
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline),
    batch_size=16
    )
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline),
    batch_size=16
    )

model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b4'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1792,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))