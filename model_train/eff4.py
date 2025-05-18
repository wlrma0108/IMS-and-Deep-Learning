import os 
import time 
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import segmentation_models_pytorch as smp
from tqdm import tqdm

# --------------------------------------------------------------------------------
# 1) .npy 파일 로드 & 전처리
# --------------------------------------------------------------------------------
prefix = r"C:\\Users\\hojoo\\Downloads\\covid-segmentation"  

images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
masks_radiopedia  = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
images_medseg     = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
masks_medseg      = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)

print("radiopedia images:", images_radiopedia.shape, 
      "radiopedia masks:",  masks_radiopedia.shape)
print("medseg images:", images_medseg.shape, 
      "medseg masks:",  masks_medseg.shape)

# convert one-hot to mask indices
masks_radiopedia_recover = np.argmax(masks_radiopedia, axis=-1)
masks_medseg_recover     = np.argmax(masks_medseg, axis=-1)

# CT HU clipping & z-score normalization
def preprocess_images(images, mean_std=None):
    images = np.clip(images, -1500, 500)
    p5, p95 = np.percentile(images, [5,95])
    valid = images[(images>=p5)&(images<=p95)]
    if mean_std is None:
        mean, std = valid.mean(), valid.std()
    else:
        mean, std = mean_std
    return (images-mean)/std, (mean, std)

images_radiopedia, mean_std = preprocess_images(images_radiopedia)
images_medseg, _ = preprocess_images(images_medseg, mean_std)

# Train/Val split
val_idx   = list(range(24))
train_idx = list(range(24, len(images_medseg)))

train_images = np.concatenate([images_medseg[train_idx], images_radiopedia],axis=0)
train_masks  = np.concatenate([masks_medseg_recover[train_idx], masks_radiopedia_recover],axis=0)
val_images   = images_medseg[val_idx]
val_masks    = masks_medseg_recover[val_idx]

print(f"[Train] images: {train_images.shape}, masks: {train_masks.shape}")
print(f"[Val]   images: {val_images.shape}, masks: {val_masks.shape}")

del images_radiopedia, masks_radiopedia, images_medseg, masks_medseg

def one_cycle_transform(img, mask, size=384):
    # strong augment
    angle = random.uniform(-30,30)
    img, mask = TF.rotate(img,angle), TF.rotate(mask,angle)
    if random.random()<0.5:
        img = TF.adjust_brightness(img, random.uniform(0.7,1.3))
        img = TF.adjust_contrast(img, random.uniform(0.7,1.3))
    if random.random()<0.5:
        img, mask = TF.hflip(img), TF.hflip(mask)
    img = TF.center_crop(img,size)
    mask = TF.center_crop(mask,size)
    return img, mask

class SegDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images, self.masks = images, masks
        self.transform = transform
        self.mean = [0.485,0.456,0.406]
        self.std  = [0.229,0.224,0.225]
    def __len__(self): return len(self.images)
    
    def __getitem__(self,idx):
        img, mask = self.images[idx], self.masks[idx]

        # 흑백(2D) 이미지를 RGB 3채널로 변환
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img, mode='RGB')

        # 마스크는 단일 채널 uint8로 변환하여 PIL Image로 생성
        mask_uint8 = mask.astype(np.uint8)
        pil_mask   = Image.fromarray(mask_uint8, mode='L')

        # 증강(transform) 적용
        if self.transform:
            pil_img, pil_mask = self.transform(pil_img, pil_mask)

        # Tensor 변환 및 정규화
        t_img = T.ToTensor()(pil_img)
        t_img = T.Normalize(self.mean, self.std)(t_img)

        # PIL 마스크를 numpy 배열로 → 2D로 보장 → LongTensor로 변환
        mask_arr = np.array(pil_mask)
        # 경우에 따라 (H,W,1) 형태가 되면 채널 차원을 제거
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        t_mask = torch.from_numpy(mask_arr).long()

        return t_img, t_mask

batch_size=2
train_ds = SegDataset(train_images,train_masks,transform=one_cycle_transform)
val_ds   = SegDataset(val_images,val_masks,transform=lambda i,m: (TF.resize(i,(384,384)),TF.resize(m,(384,384))))
train_dl = DataLoader(train_ds,batch_size,shuffle=True)
val_dl   = DataLoader(val_ds,batch_size,shuffle=False)

# metrics: accuracy, mIoU, recall, precision, f1

def pixel_accuracy(logits,mask):
    pred = logits.argmax(1)
    return (pred==mask).float().mean().item()

def mIoU(logits,mask,classes=4):
    pred=logits.argmax(1).view(-1); true=mask.view(-1)
    ious=[]
    for c in range(classes):
        pred_c=(pred==c); true_c=(true==c)
        if true_c.sum()==0: continue
        intersect=(pred_c&true_c).sum().item()
        union=(pred_c|true_c).sum().item()
        ious.append(intersect/union)
    return np.mean(ious)

def precision_recall_f1(logits,mask,classes=4):
    pred=logits.argmax(1).view(-1); true=mask.view(-1)
    precisions, recalls = [], []
    for c in range(classes):
        tp = ((pred==c)&(true==c)).sum().item()
        fp = ((pred==c)&(true!=c)).sum().item()
        fn = ((pred!=c)&(true==c)).sum().item()
        if tp+fp>0: precisions.append(tp/(tp+fp))
        if tp+fn>0: recalls.append(tp/(tp+fn))
    prec = np.mean(precisions) if precisions else 0
    rec  = np.mean(recalls)    if recalls    else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    return prec, rec, f1

# model & training
model = smp.Unet('efficientnet-b4',in_channels=3,classes=4,activation=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,1e-3,epochs=10,steps_per_epoch=len(train_dl))

def fit(epochs):
    best_loss=1e9
    for e in range(epochs):
        model.train()
        stats = {'loss':0,'acc':0,'iou':0,'prec':0,'rec':0,'f1':0}
        for x,y in tqdm(train_dl,desc=f'Train {e+1}'):
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad(); out=model(x)
            loss=criterion(out,y); loss.backward(); optimizer.step(); scheduler.step()
            stats['loss'] += loss.item()
            stats['acc']  += pixel_accuracy(out,y)
            stats['iou']  += mIoU(out,y)
            p,r,f = precision_recall_f1(out,y)
            stats['prec']+= p; stats['rec']+= r; stats['f1']+= f
        for k in stats: stats[k]/=len(train_dl)

        model.eval()
        val_stats = {'loss':0,'acc':0,'iou':0,'prec':0,'rec':0,'f1':0}
        with torch.no_grad():
            for x,y in tqdm(val_dl,desc=f'Val {e+1}'):
                x,y = x.to(device),y.to(device)
                out=model(x); loss=criterion(out,y)
                val_stats['loss']+=loss.item()
                val_stats['acc'] += pixel_accuracy(out,y)
                val_stats['iou'] += mIoU(out,y)
                p,r,f = precision_recall_f1(out,y)
                val_stats['prec']+=p; val_stats['rec']+=r; val_stats['f1']+=f
        for k in val_stats: val_stats[k]/=len(val_dl)

        print(f"Epoch {e+1}/{epochs}")
        print(f" Train Loss={stats['loss']:.3f}, Acc={stats['acc']:.3f}, mIoU={stats['iou']:.3f}, Prec={stats['prec']:.3f}, Rec={stats['rec']:.3f}, F1={stats['f1']:.3f}")
        print(f" Val   Loss={val_stats['loss']:.3f}, Acc={val_stats['acc']:.3f}, mIoU={val_stats['iou']:.3f}, Prec={val_stats['prec']:.3f}, Rec={val_stats['rec']:.3f}, F1={val_stats['f1']:.3f}\n")

        if val_stats['loss']<best_loss:
            best_loss=val_stats['loss']
            torch.save(model.state_dict(),f"best_model_epoch{e+1}.pth")
        else:
            pass

fit(10)

