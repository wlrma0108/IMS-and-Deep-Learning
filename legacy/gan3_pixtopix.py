import os, random, cv2, numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# ── 설정 ───────────────────────────────────────────────────────────
IMAGE_DIR   = r'C:\Users\hojoo\Downloads\dataset2\frames'
MASK_DIR    = r'C:\Users\hojoo\Downloads\dataset2\masks'
OUTPUT_DIR  = r'C:\Users\hojoo\Downloads\dataset2\gan_output1'
IMG_SIZE    = 256
BATCH_SIZE  = 4
EPOCHS      = 50
LR          = 2e-4
L1_LAMBDA   = 100
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED        = 42

# ── 시드 고정 ─────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

# ── 데이터셋 ─────────────────────────────────────────────────────
class Pix2PixDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        # 공통 파일만
        files = sorted(set(os.listdir(img_dir)) & set(os.listdir(mask_dir)))
        if not files:
            raise RuntimeError('No matching files in frames and masks directories')
        self.files = files
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # RGB 이미지: [-1,1]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 127.5 - 1.0
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        # 마스크: [0,1]
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return mask, img

# ── Generator (U-Net) ───────────────────────────────────────────
class UNetGenerator(nn.Module):
    def __init__(self, in_c=1, out_c=3, feat=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, feat, 4,2,1, bias=False), nn.LeakyReLU(0.2,True))
        self.enc2 = nn.Sequential(nn.Conv2d(feat, feat*2,4,2,1,bias=False), nn.InstanceNorm2d(feat*2), nn.LeakyReLU(0.2,True))
        self.enc3 = nn.Sequential(nn.Conv2d(feat*2, feat*4,4,2,1,bias=False), nn.InstanceNorm2d(feat*4), nn.LeakyReLU(0.2,True))
        self.enc4 = nn.Sequential(nn.Conv2d(feat*4, feat*8,4,2,1,bias=False), nn.InstanceNorm2d(feat*8), nn.LeakyReLU(0.2,True))
        self.bottleneck = nn.Sequential(nn.Conv2d(feat*8, feat*8,4,2,1,bias=False), nn.ReLU(True))
        # Decoder
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(feat*8, feat*8,4,2,1,bias=False), nn.InstanceNorm2d(feat*8), nn.ReLU(True), nn.Dropout(0.5))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(feat*16, feat*4,4,2,1,bias=False), nn.InstanceNorm2d(feat*4), nn.ReLU(True), nn.Dropout(0.5))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(feat*8, feat*2,4,2,1,bias=False), nn.InstanceNorm2d(feat*2), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(feat*4, feat,4,2,1,bias=False), nn.InstanceNorm2d(feat), nn.ReLU(True))
        self.out  = nn.Sequential(nn.ConvTranspose2d(feat*2, out_c,4,2,1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4],1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3],1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2],1)
        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1],1)
        return self.out(d1)

# ── Discriminator (70×70 PatchGAN) ─────────────────────────────
class PatchDiscriminator(nn.Module):
    def __init__(self, in_c=4, feat=64):
        super().__init__()
        layers = []
        dims = [feat, feat*2, feat*4, feat*8]
        for i, d in enumerate(dims):
            inc = in_c if i==0 else dims[i-1]
            stride = 1 if i==3 else 2
            layers.append(nn.Conv2d(inc, d,4,stride,1,bias=False))
            if i>0: layers.append(nn.InstanceNorm2d(d))
            layers.append(nn.LeakyReLU(0.2,True))
        layers.append(nn.Conv2d(dims[-1],1,4,1,1,bias=False))
        self.model = nn.Sequential(*layers)
    def forward(self, m, i): return self.model(torch.cat([m,i],1))

# ── 가중치 초기화 ─────────────────────────────────────────────────
def init_weights(net):
    for m in net.modules():
        # Conv layers
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        # InstanceNorm layers (affine=False may have no params)
        elif isinstance(m, nn.InstanceNorm2d):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

# ── 학습 함수 ───────────────────────────────────────────────────── ─────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # DataLoader
    dataset = Pix2PixDataset(IMAGE_DIR, MASK_DIR)
    train_size = int(0.8*len(dataset)); val_size = len(dataset)-train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,num_workers=0, pin_memory=True)
    # Models, Loss, Optimizer
    G = UNetGenerator().to(DEVICE); D = PatchDiscriminator().to(DEVICE)
    init_weights(G); init_weights(D)
    adv_loss = nn.BCEWithLogitsLoss(); l1_loss = nn.L1Loss()
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda')

    for epoch in range(1, EPOCHS+1):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
        for mask, img in pbar:
            mask, img = mask.to(DEVICE), img.to(DEVICE)
            valid = torch.ones(mask.size(0),1,30,30, device=DEVICE)
            fake  = torch.zeros_like(valid)
            # Discriminator step
            with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):
                gen_img = G(mask).detach()
                d_real = adv_loss(D(mask,img), valid)
                d_fake = adv_loss(D(mask,gen_img), fake)
                d_loss = 0.5*(d_real + d_fake)
            opt_D.zero_grad(); scaler.scale(d_loss).backward(); scaler.step(opt_D)

            # Generator step
            with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):
                gen_img = G(mask)
                g_adv = adv_loss(D(mask,gen_img), valid)
                g_l1 = l1_loss(gen_img, img)*L1_LAMBDA
                g_loss = g_adv + g_l1
            opt_G.zero_grad(); scaler.scale(g_loss).backward(); scaler.step(opt_G); scaler.update()
            pbar.set_postfix(D=f'{d_loss.item():.4f}', G=f'{g_loss.item():.4f}')

        # Save sample
        save_image(mask[0], os.path.join(OUTPUT_DIR,f'mask_{epoch:03d}.png'), normalize=True)
        save_image((gen_img[0]+1)/2, os.path.join(OUTPUT_DIR,f'gen_{epoch:03d}.png'), normalize=True)
        # Validation L1
        G.eval(); val_l1=0
        with torch.no_grad():
            for mask, img in val_loader:
                mask, img = mask.to(DEVICE), img.to(DEVICE)
                pred = G(mask)
                val_l1 += l1_loss(pred, img).item()*mask.size(0)
        val_l1 /= len(val_loader.dataset)
        print(f'Val L1: {val_l1:.4f}')

    torch.save(G.state_dict(), os.path.join(OUTPUT_DIR,'G.pth'))
    torch.save(D.state_dict(), os.path.join(OUTPUT_DIR,'D.pth'))

# ── 엔트리 포인트 ─────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(SEED)
    train()
