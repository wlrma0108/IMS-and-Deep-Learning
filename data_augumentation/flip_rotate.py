import os
import random
from glob import glob
import cv2

# 원본 폴더 설정
frame_dir = r'C:/Users/hojoo/Downloads/dataset2/frames'
mask_dir  = r'C:/Users/hojoo/Downloads/dataset2/masks'

# 증강된 결과를 저장할 폴더
aug_frame_dir = r'dataset2_FlipRotate'
aug_mask_dir  = r'dataset2_FlipRotate_masks'
os.makedirs(aug_frame_dir, exist_ok=True)
os.makedirs(aug_mask_dir,  exist_ok=True)

# 회전 코드(OpenCV가 지원하는 90° 단위 회전)
rot_codes = [
    None,                               # 회전 안 함
    cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE
]

# 프레임과 마스크 파일 리스트 (정렬해서 순서 보장)
frame_paths = sorted(glob(os.path.join(frame_dir, '*.*')))
mask_paths  = sorted(glob(os.path.join(mask_dir,  '*.*')))

assert len(frame_paths) == len(mask_paths), "프레임과 마스크 개수가 달라요!"

for idx, (fpath, mpath) in enumerate(zip(frame_paths, mask_paths)):
    # 1) 원본 로드
    img  = cv2.imread(fpath, cv2.IMREAD_COLOR)
    mask = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)  # 마스크는 채널 보존

    # 2) 랜덤 증강 파라미터 생성 (한 번만)
    rot_code = random.choice(rot_codes)
    do_hflip = (random.random() < 0.5)
    do_vflip = (random.random() < 0.5)

    # 3) 이미지와 마스크에 동일한 순서로 변환 적용
    if rot_code is not None:
        img  = cv2.rotate(img,  rot_code)
        mask = cv2.rotate(mask, rot_code)

    if do_hflip:
        img  = cv2.flip(img, 1)
        mask = cv2.flip(mask,1)

    if do_vflip:
        img  = cv2.flip(img, 0)
        mask = cv2.flip(mask,0)

    # 4) 결과 저장
    fname = os.path.basename(fpath)
    cv2.imwrite(os.path.join(aug_frame_dir, fname), img)
    cv2.imwrite(os.path.join(aug_mask_dir,  fname), mask)

print('Flip/Rotate 증강 완료 — 항상 이미지와 마스크 정합 유지됩니다.')
