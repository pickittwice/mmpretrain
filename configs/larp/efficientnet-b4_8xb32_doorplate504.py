_base_ = [
    '../_base_/models/efficientnet_b4.py',
    '../_base_/datasets/larp_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(
    dataset=dict(
        ann_file='meta/doorplate504_ann_train.json',
    ),
    batch_size=16
    )
val_dataloader = dict(
    dataset=dict(
        ann_file='meta/doorplate504_ann_val.json',
    ),
    batch_size=16
    )
test_dataloader = dict(
    dataset=dict(
        ann_file='meta/doorplate504_ann_val.json',
    ),
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