import os
from glob import glob
import cv2

input_dir  = r'C:\Users\hojoo\Desktop\ims\lena'
output_dir = r'lena'
os.makedirs(output_dir, exist_ok=True)

quality_levels = [30]   # 원하는 JPEG 품질

for path in glob(os.path.join(input_dir, '*.*')):
    img   = cv2.imread(path)
    fname = os.path.splitext(os.path.basename(path))[0]

    for q in quality_levels:
        out_path = os.path.join(output_dir, f'{fname}_q{q}.jpg')
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, q])

print('JPEG compression augmentation 완료')
