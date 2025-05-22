import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── 설정 ──
IMAGE_DIR  = r'C:\Users\kim\Desktop\ims\dataset2\frames'
MASK_DIR   = r'C:\Users\kim\Desktop\ims\dataset2\masks'
IMG_SIZE   = 512
BATCH_SIZE = 2
EPOCHS     = 5
LR         = 2e-4
L1_LAMBDA  = 100
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ──
class CovidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, files, img_size):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.files     = files
        self.img_size  = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # 이미지 로드 및 전처리
        img = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        # 마스크 로드 및 전처리
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(img), torch.from_numpy(mask)

# ── Generator (U-Net) ──
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3,64);    self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64,128);  self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128,256); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256,512); self.pool4 = nn.MaxPool2d(2)
        self.bridge = ConvBlock(512,1024)
        self.up1 = nn.ConvTranspose2d(1024,512,2,stride=2); self.dec1 = ConvBlock(1024,512)
        self.up2 = nn.ConvTranspose2d(512,256,2,stride=2);  self.dec2 = ConvBlock(512,256)
        self.up3 = nn.ConvTranspose2d(256,128,2,stride=2);  self.dec3 = ConvBlock(256,128)
        self.up4 = nn.ConvTranspose2d(128,64,2,stride=2);   self.dec4 = ConvBlock(128,64)
        self.final = nn.Conv2d(64,1,1)

    def forward(self, x):
        s1 = self.enc1(x); p1 = self.pool1(s1)
        s2 = self.enc2(p1); p2 = self.pool2(s2)
        s3 = self.enc3(p2); p3 = self.pool3(s3)
        s4 = self.enc4(p3); p4 = self.pool4(s4)
        b  = self.bridge(p4)
        d1 = torch.cat([self.up1(b), s4], dim=1); d1 = self.dec1(d1)
        d2 = torch.cat([self.up2(d1), s3], dim=1); d2 = self.dec2(d2)
        d3 = torch.cat([self.up3(d2), s2], dim=1); d3 = self.dec3(d3)
        d4 = torch.cat([self.up4(d3), s1], dim=1); d4 = self.dec4(d4)
        return torch.sigmoid(self.final(d4))

# ── Discriminator (PatchGAN) ──
class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=4):
        super().__init__()
        def conv_block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.net = nn.Sequential(
            conv_block(in_c, 64, 2),
            conv_block(64,128, 2),
            conv_block(128,256,2),
            conv_block(256,512,1),
            nn.Conv2d(512,1,4,padding=1),
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.net(x)

# ── 학습 루프 ──
def train():
    files = os.listdir(IMAGE_DIR)
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    train_ds = CovidDataset(IMAGE_DIR, MASK_DIR, train_files, IMG_SIZE)
    val_ds   = CovidDataset(IMAGE_DIR, MASK_DIR, val_files, IMG_SIZE)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    G = UNetGenerator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss  = nn.L1Loss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))

    for epoch in range(1, EPOCHS+1):
        G.train(); D.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            real_pred = D(imgs, masks)
            real_label = torch.ones_like(real_pred, device=DEVICE)
            loss_real = adv_loss(real_pred, real_label)

            fake_masks = G(imgs)
            fake_pred = D(imgs, fake_masks.detach())
            fake_label = torch.zeros_like(fake_pred, device=DEVICE)
            loss_fake = adv_loss(fake_pred, fake_label)
            # Discriminator 업데이트
            opt_D.zero_grad()
            real_pred = D(imgs, masks)
            loss_real = adv_loss(real_pred, real_label)
            fake_masks = G(imgs)
            fake_pred = D(imgs, fake_masks.detach())
            loss_fake = adv_loss(fake_pred, fake_label)
            d_loss = 0.5 * (loss_real + loss_fake)
            d_loss.backward(); opt_D.step()

            # Generator 업데이트
            opt_G.zero_grad()
            fake_pred = D(imgs, fake_masks)
            # real_pred 크기와 동일한 레이블 사용
            adv_target = torch.ones_like(fake_pred, device=DEVICE)
            g_adv = adv_loss(fake_pred, adv_target)

            g_l1  = l1_loss(fake_masks, masks) * L1_LAMBDA
            g_loss = g_adv + g_l1
            g_loss.backward(); opt_G.step()

        # Validation L1
        G.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = G(imgs)
                val_l1 += l1_loss(preds, masks).item() * imgs.size(0)
        val_l1 /= len(val_loader.dataset)

        print(f'Epoch {epoch}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Val L1: {val_l1:.4f}')

    # 모델 저장
    torch.save(G.state_dict(), 'generator.pth')
    torch.save(D.state_dict(), 'discriminator.pth')

if __name__ == '__main__':
    train()
