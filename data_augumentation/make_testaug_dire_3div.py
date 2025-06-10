import os
import shutil
from glob import glob

# 원본 디렉토리 경로
dir1 = r'C:\Users\admin\ims\dataset2\masks'
dir2 = r'C:\Users\admin\ims\dataset2\gan_mask'
dir3 = r'C:\Users\admin\ims\dataset2_aug\dataset2_elastic_masks'

# 새로 만들 디렉토리
out_dir = r'C:\Users\admin\ims\origin_and_gan_and_elastic_masks'
os.makedirs(out_dir, exist_ok=True)

# 파일 확장자에 맞게 패턴 수정 (예: '*.png' 또는 '*.*')
pattern = '*.*'

# 1. 첫 번째 디렉토리에서 앞에서 834장 (0–833)
files1 = sorted(glob(os.path.join(dir1, pattern)))
selected1 = files1[:834]

# 2. 두 번째 디렉토리에서 834장부터 1667장 전까지 (834–1667)
files2 = sorted(glob(os.path.join(dir2, pattern)))
selected2 = files2[834:1667]

# 3. 세 번째 디렉토리에서 1667장부터 2500장 전까지 (1667–2499)
files3 = sorted(glob(os.path.join(dir3, pattern)))
selected3 = files3[1667:2500]

# 3. 순서대로 복사
for src in selected1 + selected2 + selected3:
    fname = os.path.basename(src)
    dst = os.path.join(out_dir, fname)
    shutil.copy2(src, dst)

print(f"완료: {len(selected1)} + {len(selected2)} + {len(selected3)} = {len(selected1) + len(selected2) + len(selected3)} files copied to {out_dir}")
