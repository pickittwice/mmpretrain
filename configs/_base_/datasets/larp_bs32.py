# dataset settings
dataset_type = 'BaseDataset'
data_preprocessor = dict(
    num_classes=2,
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=False,
)

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
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/larp',
        ann_file='meta/playground_ann_train.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/larp',
        ann_file='meta/playground_ann_val.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = [
    dict(type='Accuracy'),
    dict(type='SingleLabelMetric', num_classes=2)
]

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
