import os
from glob import glob
import cv2
import numpy as np

input_dir  = r'C:/Users/hojoo/Downloads/dataset2/masks'
output_dir = r'dataset2_elastic_masks'
os.makedirs(output_dir, exist_ok=True)

alpha = 1.0      # 변형 강도 (픽셀 단위)
sigma = 8        # 가우시안 블러 커널 표준편차

def elastic_distort(img, alpha=1.0, sigma=8):
    """OpenCV만으로 구현한 Elastic Transform (B. Graham 알고리즘 변형)"""
    h, w = img.shape[:2]

    # 1) 무작위 변위장 생성
    dx = (np.random.rand(h, w).astype(np.float32) * 2 - 1)
    dy = (np.random.rand(h, w).astype(np.float32) * 2 - 1)

    # 2) 가우시안 블러로 변위 smooth
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    # 3) 좌표 매핑
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

for path in glob(os.path.join(input_dir, '*.*')):
    img = cv2.imread(path)
    aug = elastic_distort(img, alpha=alpha, sigma=sigma)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(path)), aug)

print('Elastic augmentation 완료')
