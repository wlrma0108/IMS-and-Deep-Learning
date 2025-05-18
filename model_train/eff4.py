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
prefix = "C:\\Users\\admin\\dataset"

images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
masks_radiopedia  = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
images_medseg     = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
masks_medseg      = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)

print("radiopedia images:", images_radiopedia.shape, 
      "radiopedia masks:",  masks_radiopedia.shape)
print("medseg images:", images_medseg.shape, 
      "medseg masks:",  masks_medseg.shape)

def onehot_to_mask(mask_1hot):
    """(H, W, K) one-hot → (H, W) integer index"""
    return np.argmax(mask_1hot, axis=-1)

masks_radiopedia_recover = onehot_to_mask(masks_radiopedia)
masks_medseg_recover     = onehot_to_mask(masks_medseg)

# CT HU값 클리핑 & z-score 정규화
def preprocess_images(images, mean_std=None):
    images[images > 500]   = 500
    images[images < -1500] = -1500
    p5, p95 = np.percentile(images, [5,95])
    valid_vals = images[(images > p5) & (images < p95)]
    if mean_std is None:
        mean = valid_vals.mean()
        std  = valid_vals.std()
    else:
        mean, std = mean_std
    images = (images - mean)/std
    return images, (mean, std)

images_radiopedia, mean_std = preprocess_images(images_radiopedia, None)
images_medseg, _ = preprocess_images(images_medseg, mean_std)

# Train/Val 분할
val_indexes = list(range(24))
train_indexes = list(range(24, len(images_medseg)))

train_images = np.concatenate([images_medseg[train_indexes], images_radiopedia], axis=0)
train_masks  = np.concatenate([masks_medseg_recover[train_indexes], masks_radiopedia_recover], axis=0)

val_images = images_medseg[val_indexes]
val_masks  = masks_medseg_recover[val_indexes]

print("[Train] images:", train_images.shape, "masks:", train_masks.shape)
print("[Val  ] images:", val_images.shape,   "masks:", val_masks.shape)

# 메모리 회수
del images_radiopedia, masks_radiopedia
del images_medseg, masks_medseg, masks_radiopedia_recover, masks_medseg_recover

# --------------------------------------------------------------------------------
# 2) 커스텀 증강 (torchvision.transforms.functional)
# --------------------------------------------------------------------------------

'''

회전 : 위치/방향 변화에 대한 불변성 학습
밝기/대비 조절 : 조명 조건 변화에 강한 모델 학습
좌우 반전 : 구조적 다양성 증가
중앙 크롭 :크기 표준화 + 주변 여백 제거

'''
class StrongAugment:
    def __init__(self, output_size=384):
        self.output_size = output_size

    def __call__(self, img, mask):
        # 기본 회전 : -30도 ~ +30도 사이의 임의 각도로 회전시킵니다.
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        # 밝기 & 대비 조정 (이미지에만) : 50% 확률로 실행됩니다. 밝기는 70%~130% 범위 내에서 조정됩니다. *마스크에는 적용 하지 않음
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))

        # 좌우 뒤집기 : 이미지와 마스크를 50% 확률로 좌우 반전합니다.
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 중앙 크롭 : 사이즈 통일 및 배치 처리의 안정성 확보를 위해 회전, 밝기 조정 후 생긴 여백이나 잘림 방지를 위해 중앙을 기준으로 256x256으로 잘라냅니다.
        img = TF.center_crop(img, self.output_size)
        mask = TF.center_crop(mask, self.output_size)

        return img, mask

class SimpleResize:
    """검증용: 380×380 Resize"""
    def __init__(self, out_size=384):
        self.out_size = out_size
    def __call__(self, img, mask):
        img  = TF.resize(img,  (self.out_size, self.out_size), interpolation=Image.NEAREST)
        mask = TF.resize(mask, (self.out_size, self.out_size), interpolation=Image.NEAREST)
        return img, mask

train_transform = StrongAugment(output_size=384)
val_transform   = SimpleResize(out_size=384)

# --------------------------------------------------------------------------------
# 3) Dataset & DataLoader
# --------------------------------------------------------------------------------
"""
    - images: (N,H,W) or (N,H,W,1)
    - masks:  (N,H,W)
    - transform: (pil_img, pil_mask)->(pil_img, pil_mask)
"""
class MySegDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images    = images
        self.masks     = masks
        self.transform = transform
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] # (H,W) or (H,W,1)
        mask  = self.masks[idx] # (H,W)

        # (H,W,1) -> (H,W)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        # 흑백 -> RGB 3채널
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        # float -> 0~255 uint8
        # (이미 -2~+2 등 범위일 수 있으므로 clip)
        image = (image * 255).clip(0,255).astype(np.uint8)
        mask  = mask.astype(np.int32)

        # NumPy -> PIL
        pil_img  = Image.fromarray(image, mode='RGB')
        pil_mask = Image.fromarray(mask,  mode='I')

        # transform
        if self.transform:
            pil_img, pil_mask = self.transform(pil_img, pil_mask)

        # ToTensor & Normalize
        t_img = T.ToTensor()(pil_img) # (3,H,W)
        t_img = T.Normalize(self.mean, self.std)(t_img)

        t_mask= torch.from_numpy(np.array(pil_mask)).long() # (H,W)

        return t_img, t_mask
    
# **Batch Size를 줄여서 OOM 방지**
batch_size = 2 # (원래 len(val_images)는 너무 큼 -> GPU 메모리 초과)
train_dataset = MySegDataset(train_images, train_masks, transform=train_transform)
val_dataset   = MySegDataset(val_images,   val_masks,   transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# --------------------------------------------------------------------------------
# 4) 시각화 수정
# --------------------------------------------------------------------------------
def visualize_samples(dataloader, n=2):
    data_iter = iter(dataloader)
    imgs, msks = next(data_iter)
    print("Sample batch - imgs:", imgs.shape, imgs.dtype)  # (B,3,H,W), float
    print("Sample batch - msks:", msks.shape, msks.dtype)  # (B,H,W), long

    for i in range(min(n, len(imgs))):
        img_np  = imgs[i].cpu().numpy().transpose(1,2,0)
        mask_np = msks[i].cpu().numpy()
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].imshow(img_np.astype(np.float32))
        ax[1].imshow(mask_np, cmap='jet')
        ax[0].set_title("Image (RGB)")
        ax[1].set_title("Mask")
        plt.show()

visualize_samples(train_dataloader, n=2)

# --------------------------------------------------------------------------------
# 5) 세그멘테이션 모델 & 학습
# --------------------------------------------------------------------------------
model = smp.Unet(
    encoder_name='efficientnet-b4',
    in_channels=3,
    encoder_weights='imagenet',
    classes=4,  
    activation=None,
    decoder_dropout=0.2 # 정규화를 위한 수정
)

def pixel_accuracy(output, mask):
    with torch.no_grad():
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = (pred == mask).float().sum()
        total   = mask.numel()
        return float(correct/total)

def mIoU(logits, mask, smooth=1e-10, n_classes=4):
    with torch.no_grad():
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        pred = pred.view(-1)
        mask = mask.view(-1)
        ious = []
        for c in range(n_classes):
            pred_c = (pred == c)
            mask_c = (mask == c)
            if mask_c.long().sum().item() == 0:
                ious.append(np.nan)
            else:
                intersect = (pred_c & mask_c).sum().item()
                union     = (pred_c | mask_c).sum().item()
                iou_val   = (intersect+smooth)/(union+smooth)
                ious.append(iou_val)
        return np.nanmean(ious)

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    device = next(model.parameters()).device
    min_val_loss = float('inf')
    no_improve   = 0

    for e in range(epochs):
        model.train()
        train_loss=0.0
        train_acc =0.0
        train_iou =0.0
        torch.cuda.empty_cache()

        for imgs, msks in tqdm(train_loader, desc=f"Epoch {e+1}/{epochs} [Train]"):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, msks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_acc  += pixel_accuracy(outputs, msks)
            train_iou  += mIoU(outputs, msks, n_classes=4)

        n_train = len(train_loader)
        train_loss /= n_train
        train_acc  /= n_train
        train_iou  /= n_train

        model.eval()
        val_loss=0.0
        val_acc =0.0
        val_iou =0.0
        with torch.no_grad():
            for imgs, msks in tqdm(val_loader, desc=f"Epoch {e+1}/{epochs} [Val]"):
                imgs, msks = imgs.to(device), msks.to(device)
                logits= model(imgs)
                loss = criterion(logits, msks)
                val_loss += loss.item()
                val_acc  += pixel_accuracy(logits, msks)
                val_iou  += mIoU(logits, msks, n_classes=4)

        n_val   = len(val_loader)
        val_loss/= n_val
        val_acc /= n_val
        val_iou /= n_val

        lr = scheduler.get_last_lr()[0]
        print(f"\n[Epoch {e+1}/{epochs}]"
              f" Train Loss={train_loss:.3f}, Acc={train_acc:.3f}, mIoU={train_iou:.3f} | "
              f"Val Loss={val_loss:.3f}, Acc={val_acc:.3f}, mIoU={val_iou:.3f}, LR={lr:.6f}")

        if val_loss < min_val_loss:
            print(f"Val Loss improved {min_val_loss:.3f} -> {val_loss:.3f}. Saving model...")
            min_val_loss = val_loss
            no_improve   = 0
            torch.save(model.state_dict(), f"Unet_effb4_epoch{e+1}_val{val_loss:.3f}.pth")
        else:
            no_improve+=1
            print(f"Val Loss not improved count = {no_improve}")
            if no_improve>=7:
                print("No improvement for 7 epochs, stopping early.")
                break
# --------------------------------------------------------------------------------
# 6) 실제 학습
# --------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs=10
max_lr=1e-3
weight_decay=1e-4 

criterion=nn.CrossEntropyLoss()

# Batch Size를 줄였으므로 step_per_epoch가 많아진다
optimizer=torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
scheduler=torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dataloader)
)

fit(epochs, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler)
