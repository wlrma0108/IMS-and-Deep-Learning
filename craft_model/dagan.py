import os, random, cv2, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# ──────────────────── 설정 ────────────────────────
IMAGE_DIR  = r'C:\Users\hojoo\Desktop\ims\dataset2\frames'
MASK_DIR   = r'C:\Users\hojoo\Desktop\ims\dataset2\masks'
OUTPUT_DIR = r'C:/Users/hojoo/Downloads/dataset2/dagan_output'
IMG_SIZE   = 256
BATCH_SIZE = 4
EPOCHS     = 50
LR         = 2e-4
NOISE_DIM  = 100
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED       = 42

# ───────────────── 시드 고정 ───────────────────────
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type=='cuda': torch.cuda.manual_seed_all(seed)

# ───────────────── Dataset ─────────────────────────
class ImageMaskDataset(Dataset):
    def __init__(self, img_dir:str, mask_dir:str):
        self.files = sorted(set(os.listdir(img_dir)) & set(os.listdir(mask_dir)))
        if not self.files:
            raise RuntimeError('frames/와 masks/에 공통 파일이 없습니다.')
        self.img_dir, self.mask_dir = img_dir, mask_dir
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx:int):
        f = self.files[idx]
        # 이미지 [-1,1]
        img = cv2.imread(os.path.join(self.img_dir,f))
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/127.5 - 1.0
        img = torch.from_numpy(img.transpose(2,0,1))
        # 마스크 [0,1]
        mask = cv2.imread(os.path.join(self.mask_dir,f),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(IMG_SIZE,IMG_SIZE)).astype(np.float32)/255.0
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img.float(), mask.float()

# ───────────────── Generator ────────────────────────
class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__(); self.block = nn.Sequential(
            nn.Conv2d(ic, oc, 4,2,1,bias=False), nn.BatchNorm2d(oc), nn.LeakyReLU(0.2,True))
    def forward(self,x): return self.block(x)

class Up(nn.Module):
    def __init__(self, ic, oc):
        super().__init__(); self.block = nn.Sequential(
            nn.ConvTranspose2d(ic, oc, 4,2,1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self,x): return self.block(x)

class DAGANGenerator(nn.Module):
    """U‑Net + AdaIN(noise) : 이미지·마스크 동시 생성"""
    def __init__(self, img_c=3, mask_c=1, noise_dim=100, feat=64):
        super().__init__(); self.noise_dim=noise_dim
        # Encoder
        self.d1=Down(img_c+mask_c,feat); self.d2=Down(feat,feat*2); self.d3=Down(feat*2,feat*4); self.d4=Down(feat*4,feat*8)
        self.bottleneck = nn.Sequential(nn.Conv2d(feat*8,feat*8,4,2,1,bias=False), nn.BatchNorm2d(feat*8), nn.ReLU(True))
        # AdaIN affine
        self.fc_scale = nn.Linear(noise_dim, feat*8); self.fc_shift = nn.Linear(noise_dim, feat*8)
        # Decoder
        self.u4=Up(feat*16,feat*4); self.u3=Up(feat*8,feat*2); self.u2=Up(feat*4,feat); self.u1=Up(feat*2,feat)
        self.to_img = nn.Sequential(nn.Conv2d(feat, img_c,3,1,1), nn.Tanh())
        self.to_mask= nn.Sequential(nn.Conv2d(feat, mask_c,3,1,1), nn.Sigmoid())
    def forward(self,img,mask,z):
        x=torch.cat([img,mask],1)
        e1=self.d1(x); e2=self.d2(e1); e3=self.d3(e2); e4=self.d4(e3)
        b=self.bottleneck(e4)
        scale=self.fc_scale(z).view(-1,b.size(1),1,1); shift=self.fc_shift(z).view(-1,b.size(1),1,1)
        b = b*(scale+1)+shift
        d4=self.u4(torch.cat([b,e4],1)); d3=self.u3(torch.cat([d4,e3],1)); d2=self.u2(torch.cat([d3,e2],1)); d1=self.u1(torch.cat([d2,e1],1))
        return self.to_img(d1), self.to_mask(d1)

# ───────────────── Discriminator ─────────────────────
class DAGANDiscriminator(nn.Module):
    def __init__(self,in_c=4,feat=64):
        super().__init__(); layers=[]; dims=[feat,feat*2,feat*4,feat*8]
        for i,d in enumerate(dims):
            inc=in_c if i==0 else dims[i-1]; stride=1 if i==3 else 2
            layers+=[nn.Conv2d(inc,d,4,stride,1), nn.BatchNorm2d(d), nn.LeakyReLU(0.2,True)]
        layers.append(nn.Conv2d(dims[-1],1,4,1,1))
        self.model=nn.Sequential(*layers)
    def forward(self,pair): return self.model(pair).view(pair.size(0),-1)

# ───────────────── 학습 루프 ─────────────────────────

def train():
    set_seed(SEED); os.makedirs(OUTPUT_DIR,exist_ok=True)
    loader=DataLoader(ImageMaskDataset(IMAGE_DIR,MASK_DIR),batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
    G,D=DAGANGenerator().to(DEVICE),DAGANDiscriminator().to(DEVICE)
    opt_G=optim.Adam(G.parameters(),lr=LR,betas=(0,0.99)); opt_D=optim.Adam(D.parameters(),lr=LR,betas=(0,0.99))
    criterion=nn.BCEWithLogitsLoss()

    for epoch in range(1,EPOCHS+1):
        loop=tqdm(loader,desc=f'Epoch {epoch}/{EPOCHS}')
        for img,mask in loop:
            img,mask=img.to(DEVICE),mask.to(DEVICE); bs=img.size(0); z=torch.randn(bs,NOISE_DIM,device=DEVICE)
            fake_img,fake_mask=G(img,mask,z)
            fake_pair=torch.cat([fake_img,fake_mask],1); real_pair=torch.cat([img,mask],1)
            # Discriminator
            d_loss=0.5*(criterion(D(real_pair),torch.ones(bs,1,device=DEVICE))+criterion(D(fake_pair.detach()),torch.zeros(bs,1,device=DEVICE)))
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            # Generator
            g_loss=criterion(D(fake_pair),torch.ones(bs,1,device=DEVICE))
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            loop.set_postfix(D=f"{d_loss.item():.4f}",G=f"{g_loss.item():.4f}")
        # 샘플 저장
        with torch.no_grad():
            z=torch.randn(4,NOISE_DIM,device=DEVICE); img_s,mask_s=next(iter(loader)); img_s,mask_s=img_s[:4].to(DEVICE),mask_s[:4].to(DEVICE)
            gen_img,gen_mask=G(img_s,mask_s,z)
            save_image((gen_img+1)/2,os.path.join(OUTPUT_DIR,f'ep{epoch:03d}_img.png'))
            cv2.imwrite(os.path.join(OUTPUT_DIR,f'ep{epoch:03d}_mask.png'),(gen_mask.squeeze(1).cpu().numpy()*255).astype(np.uint8))
    torch.save(G.state_dict(),os.path.join(OUTPUT_DIR,'G.pth'))
    torch.save(D.state_dict(),os.path.join(OUTPUT_DIR,'D.pth'))

if __name__=='__main__':
    train()
