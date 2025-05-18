import os
import random
from glob import glob
import cv2
import numpy as np

input_dir  = r'C:/Users/hojoo/Downloads/dataset2/masks'
output_dir = r'dataset2_mixup_masks'
os.makedirs(output_dir, exist_ok=True)

paths = glob(os.path.join(input_dir, '*.*'))
for i in range(len(paths)):
    p1, p2 = random.sample(paths, 2)

    img1 = cv2.imread(p1).astype(np.float32) / 255.0
    img2 = cv2.imread(p2).astype(np.float32) / 255.0
    lam  = random.uniform(0.3, 0.7)

    mixed = (lam * img1 + (1 - lam) * img2)
    mixed = (mixed * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, f'mixup_{i:04d}.png'), mixed)

print('MixUp augmentation 완료')
