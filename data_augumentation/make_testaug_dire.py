import os
import shutil
from glob import glob

# 원본 디렉토리 경로
dir1 = r'C:/path/to/first_directory'
dir2 = r'C:/path/to/second_directory'

# 새로 만들 디렉토리
out_dir = r'C:/path/to/output_directory'
os.makedirs(out_dir, exist_ok=True)

# 파일 확장자에 맞게 패턴 수정 (예: '*.png' 또는 '*.*')
pattern = '*.*'

# 1. 첫 번째 디렉토리에서 앞에서 1250장 (0–1249)
files1 = sorted(glob(os.path.join(dir1, pattern)))
selected1 = files1[:1250]

# 2. 두 번째 디렉토리에서 1250장부터 2500장 전까지 (1250–2499)
files2 = sorted(glob(os.path.join(dir2, pattern)))
selected2 = files2[1250:2500]

# 3. 순서대로 복사
for src in selected1 + selected2:
    fname = os.path.basename(src)
    dst = os.path.join(out_dir, fname)
    shutil.copy2(src, dst)

print(f"완료: {len(selected1)} + {len(selected2)} = {len(selected1) + len(selected2)} files copied to {out_dir}")
