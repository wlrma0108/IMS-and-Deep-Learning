import os
import random
from glob import glob
import cv2

input_dir  = r'C:/Users/hojoo/Downloads/dataset2/masks'
output_dir = r'dataset2_FlipRotate_masks'
os.makedirs(output_dir, exist_ok=True)

# 회전 코드(OpenCV가 지원하는 90° 단위 회전)
rot_codes = [
    None,                               # 회전 안 함
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE
]

for path in glob(os.path.join(input_dir, '*.*')):
    img = cv2.imread(path)

    # 무작위 90° 회전
    rot_code = random.choice(rot_codes)
    if rot_code is not None:
        img = cv2.rotate(img, rot_code)

    # 무작위 수평/수직 플립
    if random.random() < 0.5:
        img = cv2.flip(img, 1)          # 좌–우
    if random.random() < 0.5:
        img = cv2.flip(img, 0)          # 상–하

    cv2.imwrite(os.path.join(output_dir, os.path.basename(path)), img)

print('Flip/Rotate augmentation 완료')
