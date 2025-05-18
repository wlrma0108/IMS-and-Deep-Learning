import os
import glob
import numpy as np
from monai.transforms import (
    LoadImageD, SaveImageD, Compose, 
    RandFlipD, RandRotate90D, RandElasticD, RandGaussianNoised, RandAdjustContrastD
)
from monai.data import Dataset, DataLoader

input_dir = 'train'
output_root = 'train_MONAI'
ous_dtypes = ('.png', '.jpg', '.jpeg', '.nii', '.nii.gz')

os.makedirs(output_root, exist_ok=True)

data_dicts = []
for ext in ous_dtypes:
    for path in glob.glob(os.path.join(input_dir, f'*{ext}')):
        data_dicts.append({'image': path})

augs = Compose([
    LoadImageD(keys=['image']),
    RandFlipD(keys=['image'], prob=0.5, spatial_axis=0),
    RandRotate90D(keys=['image'], prob=0.5, max_k=3),
    RandElasticD(
        keys=['image'], prob=0.3,
        sigma_range=(5, 8), magnitude_range=(100, 200), spatial_size=None,
    ),
    RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
    RandAdjustContrastD(keys=['image'], prob=0.3, gamma=(0.7, 1.5)),
])

dataset = Dataset(data=data_dicts, transform=augs)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
for idx, batch in enumerate(loader):
    img = batch['image'][0]
    save_dict = {
        'image': img,
        'meta_dict': batch['image_meta_dict'][0]
    }
    saver = Compose([
        SaveImageD(
            keys=['image'], 
            output_dir=output_root,
            output_postfix=f"aug{idx:03d}",
            resample=False,
            separate_folder=False
        )
    ])
    saver(save_dict)

print("MONAI 증강 완료: 저장 위치 ->", output_root)