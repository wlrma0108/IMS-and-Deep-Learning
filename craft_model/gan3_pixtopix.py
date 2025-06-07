import os, random, cv2, numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── 설정 ───────────────────────────────────────────────────────────
IMAGE_DIR   = r'C:\Users\hojoo\Downloads\dataset2\frames'
MASK_DIR    = r'C:\Users\hojoo\Downloads\dataset2\masks'
OUTPUT_DIR  = r'C:\Users\hojoo\Downloads\dataset2\gan_output1'
IMG_SIZE    = 256
BATCH_SIZE  = 8
EPOCHS      = 100
LR          = 2e-4
L1_LAMBDA   = 100
NUM_SAMPLES = 2500
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용중인 장치 : {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ──
class Pix2PixDataset(Dataset):
    def __init__(self, image_dir, mask_dir, files, img_size):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.files, self.img_size     = files, img_size

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # load and preprocess image
        img = cv2.imread(os.path.join(self.image_dir, fname))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        # load and preprocess mask/label
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, 0)

        return torch.from_numpy(img), torch.from_numpy(mask)

# ── Generator (U-Net) ──
class ConvBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, 3, 1, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, 1, 1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 64);    self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128);  self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512); self.pool4 = nn.MaxPool2d(2)
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

# ── Discriminator (PatchGAN) ──
class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=4):
        super().__init__()
        def block(ic, oc, stride):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 4, stride, 1),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, True)
            )

        self.net = nn.Sequential(
            block(in_c, 64, 2),
            block(64, 128, 2),
            block(128, 256, 2),
            block(256, 512, 1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], 1)
        return self.net(x)

# ── 학습 함수 ──
def train():
    # 파일 로딩 및 분할
    files = random.sample(os.listdir(IMAGE_DIR), min(NUM_SAMPLES, len(os.listdir(IMAGE_DIR))))
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        Pix2PixDataset(IMAGE_DIR, MASK_DIR, train_files, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        Pix2PixDataset(IMAGE_DIR, MASK_DIR, val_files, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # 모델, 손실, 최적화
    G = UNetGenerator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1  = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    # 학습 루프
    for epoch in range(1, EPOCHS+1):
        G.train(); D.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            # -------------------------
            #  1) Discriminator 업데이트
            # -------------------------
            opt_D.zero_grad()
            # real
            real_pred = D(imgs, masks)
            real_gt   = torch.ones_like(real_pred)
            loss_D_real = criterion_GAN(real_pred, real_gt)

            # fake
            fake_masks = G(imgs)
            fake_pred  = D(imgs, fake_masks.detach())
            fake_gt    = torch.zeros_like(fake_pred)
            loss_D_fake = criterion_GAN(fake_pred, fake_gt)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # ------------------------
            #  2) Generator 업데이트
            # ------------------------
            opt_G.zero_grad()
            fake_pred = D(imgs, fake_masks)
            # GAN loss + L1 loss
            loss_G_GAN = criterion_GAN(fake_pred, real_gt)
            loss_G_L1  = criterion_L1(fake_masks, masks) * L1_LAMBDA
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

        # 검증 L1 계산
        G.eval()
        val_l1 = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = G(imgs)
                val_l1 += criterion_L1(preds, masks).item() * imgs.size(0)
        val_l1 = val_l1 / len(val_loader.dataset)

        print(f"Epoch [{epoch}/{EPOCHS}]  loss_D: {loss_D.item():.4f}  "
              f"loss_G: {loss_G.item():.4f}  val_L1: {val_l1:.4f}")

        # 샘플 시각화 저장
        if epoch % 10 == 0:
            sample_img = imgs[0].permute(1,2,0).cpu().numpy() * 255
            sample_mask = preds[0,0].cpu().numpy() * 255
            overlay = sample_img.copy()
            overlay[sample_mask>127] = [255, 0, 0]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"overlay_epoch{epoch}.png"), overlay)

    # 모델 저장
    torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, 'generator.pth'))
    torch.save(D.state_dict(), os.path.join(OUTPUT_DIR, 'discriminator.pth'))
    print("Training complete. Models saved.")

if __name__ == '__main__':
    train()
