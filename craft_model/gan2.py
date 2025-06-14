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
EPOCHS      = 50
LR          = 2e-4
L1_LAMBDA   = 100
NUM_SAMPLES = 2500
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용중인 장치 : {DEVICE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ──
class CovidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, files, img_size):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.files, self.img_size     = files, img_size
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img  = cv2.resize(cv2.imread(os.path.join(self.image_dir, fname)), (self.img_size,self.img_size))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img  = np.transpose(img,(2,0,1))
        mask = cv2.resize(cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE),
                          (self.img_size,self.img_size)).astype(np.float32)/255.0
        mask = np.expand_dims(mask,0)
        return torch.from_numpy(img), torch.from_numpy(mask)

# ── Generator (U-Net) ──
class ConvBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc,outc,3,1,1), nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,3,1,1),nn.BatchNorm2d(outc), nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=ConvBlock(3,64);   self.pool1=nn.MaxPool2d(2)
        self.enc2=ConvBlock(64,128); self.pool2=nn.MaxPool2d(2)
        self.enc3=ConvBlock(128,256);self.pool3=nn.MaxPool2d(2)
        self.enc4=ConvBlock(256,512);self.pool4=nn.MaxPool2d(2)
        self.bridge=ConvBlock(512,1024)
        self.up1=nn.ConvTranspose2d(1024,512,2,2); self.dec1=ConvBlock(1024,512)
        self.up2=nn.ConvTranspose2d(512,256,2,2);  self.dec2=ConvBlock(512,256)
        self.up3=nn.ConvTranspose2d(256,128,2,2);  self.dec3=ConvBlock(256,128)
        self.up4=nn.ConvTranspose2d(128,64,2,2);   self.dec4=ConvBlock(128,64)
        self.final=nn.Conv2d(64,1,1)
    def forward(self,x):
        s1=self.enc1(x); p1=self.pool1(s1)
        s2=self.enc2(p1); p2=self.pool2(s2)
        s3=self.enc3(p2); p3=self.pool3(s3)
        s4=self.enc4(p3); p4=self.pool4(s4)
        b=self.bridge(p4)
        d1=self.dec1(torch.cat([self.up1(b),s4],1))
        d2=self.dec2(torch.cat([self.up2(d1),s3],1))
        d3=self.dec3(torch.cat([self.up3(d2),s2],1))
        d4=self.dec4(torch.cat([self.up4(d3),s1],1))
        return torch.sigmoid(self.final(d4))

# ── Discriminator (PatchGAN) ──
class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=4):
        super().__init__()
        def block(ic,oc,stride): 
            return nn.Sequential(nn.Conv2d(ic,oc,4,stride,1),
                                 nn.BatchNorm2d(oc), nn.LeakyReLU(0.2,True))
        self.net=nn.Sequential(
            block(in_c,64,2), block(64,128,2), block(128,256,2),
            block(256,512,1), nn.Conv2d(512,1,4,1,1))
    def forward(self,img,mask): return self.net(torch.cat([img,mask],1))

# ── 학습 함수 ──
def train():
    files = random.sample(os.listdir(IMAGE_DIR), min(NUM_SAMPLES, len(os.listdir(IMAGE_DIR))))
    tr, val = train_test_split(files,test_size=0.2,random_state=42)
    tr_loader = DataLoader(CovidDataset(IMAGE_DIR,MASK_DIR,tr, IMG_SIZE), BATCH_SIZE, shuffle=True)
    val_loader= DataLoader(CovidDataset(IMAGE_DIR,MASK_DIR,val, IMG_SIZE), BATCH_SIZE)

    G, D = UNetGenerator().to(DEVICE), PatchDiscriminator().to(DEVICE)
    adv, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    opt_G = optim.Adam(G.parameters(), LR, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), LR, betas=(0.5,0.999))

    for epoch in range(1,EPOCHS+1):
        G.train(); D.train()
        for imgs,masks in tr_loader:
            imgs,masks = imgs.to(DEVICE), masks.to(DEVICE)

            # Discriminator
            opt_D.zero_grad()
            real_out=D(imgs,masks); fake_masks=G(imgs)
            d_loss=0.5*(adv(real_out, torch.ones_like(real_out)) +
                        adv(D(imgs,fake_masks.detach()), torch.zeros_like(real_out)))
            d_loss.backward(); opt_D.step()

            # Generator
            opt_G.zero_grad()
            fake_out=D(imgs,fake_masks)
            g_loss = adv(fake_out, torch.ones_like(fake_out)) + l1(fake_masks,masks)*L1_LAMBDA
            g_loss.backward(); opt_G.step()

        # Validation L1
        G.eval(); val_l1=0
        with torch.no_grad():
            for imgs,masks in val_loader:
                imgs,masks=imgs.to(DEVICE),masks.to(DEVICE)
                val_l1+=l1(G(imgs),masks).item()*imgs.size(0)
        val_l1/=len(val_loader.dataset)
        print(f'Epoch {epoch}/{EPOCHS} | D:{d_loss.item():.4f} G:{g_loss.item():.4f} | ValL1:{val_l1:.4f}')

    # ── 예시 출력 저장 ──
    torch.save(G.state_dict(), os.path.join(OUTPUT_DIR,'generator.pth'))
    sample = val[0]
    img_bgr = cv2.resize(cv2.imread(os.path.join(IMAGE_DIR,sample)), (IMG_SIZE,IMG_SIZE))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    inp = torch.from_numpy(np.transpose(rgb,(2,0,1))[np.newaxis]).to(DEVICE)
    with torch.no_grad():
        mask_pred = (G(inp)>0.5).float().cpu().numpy()[0,0]
    cv2.imwrite(os.path.join(OUTPUT_DIR,'pred_mask.png'), (mask_pred*255).astype(np.uint8))
    over = img_bgr.copy(); over[mask_pred==1]=(0,0,255)
    cv2.imwrite(os.path.join(OUTPUT_DIR,'overlay.png'), cv2.addWeighted(img_bgr,0.7,over,0.3,0))
    print('Done. Check output/pred_mask.png & overlay.png')

if __name__ == '__main__':
    train()
