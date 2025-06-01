import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.utils import save_image

# ── U-Net Generator 정의 ──
class ConvBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, 3, 1, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, 1, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 64);   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256);self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512);self.pool4 = nn.MaxPool2d(2)
        self.bridge = ConvBlock(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2); self.dec1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2);  self.dec2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2);  self.dec3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2);   self.dec4 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        s1 = self.enc1(x); p1 = self.pool1(s1)
        s2 = self.enc2(p1); p2 = self.pool2(s2)
        s3 = self.enc3(p2); p3 = self.pool3(s3)
        s4 = self.enc4(p3); p4 = self.pool4(s4)
        b = self.bridge(p4)
        d1 = self.dec1(torch.cat([self.up1(b), s4], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), s3], 1))
        d3 = self.dec3(torch.cat([self.up3(d2), s2], 1))
        d4 = self.dec4(torch.cat([self.up4(d3), s1], 1))
        return torch.sigmoid(self.final(d4))

# ── 환경설정 ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
FRAME_DIR = r'C:\Users\hojoo\Downloads\dataset2\frames'
OUTPUT_FRAME_DIR = 'gan_frame'
OUTPUT_MASK_DIR = 'gan_mask'
os.makedirs(OUTPUT_FRAME_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# ── Generator 로드 ──
gen = UNetGenerator().to(DEVICE)
checkpoint = torch.load(r'C:\Users\hojoo\Downloads\dataset2\gan_output1\generator.pth', map_location=DEVICE)
gen.load_state_dict(checkpoint)
gen.eval()

# ── 이미지 불러와서 마스크 생성 및 저장 ──
frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
with torch.no_grad():
    for idx, fname in enumerate(frame_files):
        img_path = os.path.join(FRAME_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"이미지 로드 실패: {img_path}")
            continue
        img = cv2.cvtColor(cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = torch.from_numpy(np.transpose(img, (2, 0, 1))[np.newaxis]).to(DEVICE)
        mask_pred = gen(inp)
        mask_pred = (mask_pred > 0.5).float()

        # 저장
        frame_save_path = os.path.join(OUTPUT_FRAME_DIR, f"frame_{idx:04d}.png")
        mask_save_path  = os.path.join(OUTPUT_MASK_DIR,  f"mask_{idx:04d}.png")
        save_image(torch.from_numpy(np.transpose(img, (2, 0, 1))), frame_save_path)
        save_image(mask_pred[0], mask_save_path)

print(f"총 {len(frame_files)}개의 이미지(frame)와 마스크(mask)를 생성하여 각각 '{OUTPUT_FRAME_DIR}' 폴더와 '{OUTPUT_MASK_DIR}' 폴더에 저장했습니다.")