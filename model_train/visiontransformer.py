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
prefix = "./dataset"  # 실제 경로에 맞춰 수정

# (1) 라디오피디아 & MedSeg .npy 로드
images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
masks_radiopedia  = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
images_medseg     = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
masks_medseg      = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)

print(smp.encoders.get_encoder_names())
print("radiopedia images:", images_radiopedia.shape, 
      "radiopedia masks:",  masks_radiopedia.shape)
print("medseg images:", images_medseg.shape, 
      "medseg masks:",  masks_medseg.shape)

# (2) (H,W,K) → (H,W) : one-hot 마스크를 인덱스 맵으로
def onehot_to_mask(mask_1hot):
    """(H, W, K) one-hot → (H, W) integer index"""
    return np.argmax(mask_1hot, axis=-1)

masks_radiopedia_recover = onehot_to_mask(masks_radiopedia)
masks_medseg_recover     = onehot_to_mask(masks_medseg)

# (3) CT HU값 클리핑 & z-score 정규화
def preprocess_images(images, mean_std=None):
    """
    1) HU를 [-1500, 500] 범위로 클리핑
    2) 5~95 분위수 구간으로 평균/표준편차 산출
    3) (x - mean)/std
    """
    images[images > 500]   = 500
    images[images < -1500] = -1500
    p5, p95   = np.percentile(images, [5,95])
    valid_vals= images[(images > p5) & (images < p95)]
    if mean_std is None:
        mean = valid_vals.mean()
        std  = valid_vals.std()
    else:
        mean, std = mean_std

    images = (images - mean)/std
    return images, (mean, std)

# 라디오피디아 -> mean_std 계산
images_radiopedia, mean_std = preprocess_images(images_radiopedia, None)
# MedSeg -> 동일 (mean, std)로 정규화
images_medseg,   _         = preprocess_images(images_medseg, mean_std)

# (4) 학습/검증 분할
val_indexes   = list(range(24))                           
train_indexes = list(range(24, len(images_medseg)))       

train_images = np.concatenate([images_medseg[train_indexes], images_radiopedia], axis=0)
train_masks  = np.concatenate([masks_medseg_recover[train_indexes], masks_radiopedia_recover], axis=0)

val_images = images_medseg[val_indexes]
val_masks  = masks_medseg_recover[val_indexes]

print("[Train] images:", train_images.shape, "masks:", train_masks.shape)
print("[Val  ] images:", val_images.shape,   "masks:", val_masks.shape)

# 메모리 절약
del images_radiopedia, masks_radiopedia
del images_medseg, masks_medseg
del masks_radiopedia_recover, masks_medseg_recover

# --------------------------------------------------------------------------------
# 2) 커스텀 증강 (torchvision.transforms.functional)
# --------------------------------------------------------------------------------
class RandomRotationCropFlip:
    """
    - ±180도 범위 내 Random Rotation
    - (224 x 224) Random Crop
    - 0.5 확률로 좌우 뒤집기
    """
    def __init__(self, output_size=224, rotation_degree=180, p_flip=0.5):
        self.output_size = output_size
        self.degree      = rotation_degree
        self.p_flip      = p_flip

    def __call__(self, img, mask):
        # (1) 랜덤 회전
        angle = random.uniform(-self.degree, self.degree)
        img   = TF.rotate(img, angle=angle, fill=0)
        mask  = TF.rotate(mask, angle=angle, fill=0)

        # (2) 랜덤 크롭 or 중앙 크롭 (224x224)
        w, h   = img.size
        crop_h = crop_w = self.output_size
        if (w > crop_w) and (h > crop_h):
            left = random.randint(0, w - crop_w)
            top  = random.randint(0, h - crop_h)
            img  = TF.crop(img, top, left, crop_h, crop_w)
            mask = TF.crop(mask, top, left, crop_h, crop_w)
        else:
            # 작으면 중앙 크롭
            left = max(0, (w - crop_w)//2)
            top  = max(0, (h - crop_h)//2)
            cw   = min(w, crop_w)
            ch   = min(h, crop_h)
            img  = TF.crop(img, top, left, ch, cw)
            mask = TF.crop(mask, top, left, ch, cw)

        # (3) 좌우 뒤집기
        if random.random() < self.p_flip:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        return img, mask

class SimpleResize:
    """검증용: 224×224 리사이즈"""
    def __init__(self, out_size=224):
        self.out_size = out_size
    def __call__(self, img, mask):
        img  = TF.resize(img,  (self.out_size, self.out_size), interpolation=Image.NEAREST)
        mask = TF.resize(mask, (self.out_size, self.out_size), interpolation=Image.NEAREST)
        return img, mask

train_transform = RandomRotationCropFlip(output_size=224)
val_transform   = SimpleResize(out_size=224)

# --------------------------------------------------------------------------------
# 3) Dataset & DataLoader
# --------------------------------------------------------------------------------
class MySegDataset(Dataset):
    """
    - images: (N,H,W) float
    - masks:  (N,H,W) int
    - transform: (pil_img, pil_mask)->(pil_img, pil_mask)
    """
    def __init__(self, images, masks, transform=None):
        self.images    = images
        self.masks     = masks
        self.transform = transform
        # ImageNet 통계 (RGB 3채널)
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask  = self.masks[idx]

        # (H,W,1) -> (H,W)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)

        # 흑백 -> RGB 3채널
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        # float -> [0,255] uint8
        image = (image * 255).clip(0,255).astype(np.uint8)
        mask  = mask.astype(np.int32)

        # PIL 변환
        pil_img  = Image.fromarray(image, mode='RGB')
        pil_mask = Image.fromarray(mask,  mode='I')

        # transform
        if self.transform:
            pil_img, pil_mask = self.transform(pil_img, pil_mask)

        # ToTensor & Normalize
        t_img = T.ToTensor()(pil_img)  # (3,H,W)
        t_img = T.Normalize(self.mean, self.std)(t_img)
        t_mask= torch.from_numpy(np.array(pil_mask)).long()

        return t_img, t_mask

batch_size = 2
train_dataset = MySegDataset(train_images, train_masks, transform=train_transform)
val_dataset   = MySegDataset(val_images,   val_masks,   transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# --------------------------------------------------------------------------------
# 4) 시각화 함수 (개선)
# --------------------------------------------------------------------------------
def visualize_samples(dataloader, n=2):
    """
    한 번의 배치에서 n개 샘플을 시각화.
    - 이미지는 (3,H,W)->(H,W,3)로 변형
    - 마스크는 cmap='jet' + colorbar 추가
    """
    data_iter = iter(dataloader)
    imgs, msks = next(data_iter)

    print("[Visualization] Batch Shape - imgs:", imgs.shape, imgs.dtype)
    print("[Visualization] Batch Shape - msks:", msks.shape, msks.dtype)

    for i in range(min(n, len(imgs))):
        # Tensor -> NumPy, (C,H,W)->(H,W,C)
        img_np  = imgs[i].cpu().numpy().transpose(1,2,0)
        mask_np = msks[i].cpu().numpy()

        # 시각적 용도로 float32 -> float64 변환(precision)
        img_np  = img_np.astype(np.float64)

        # 준비
        fig, ax = plt.subplots(1,2, figsize=(10,5))

        # 첫 번째 subplot - 원본 이미지
        ax[0].imshow(img_np)
        ax[0].set_title(f"Image {i} (RGB)")
        ax[0].axis('off')

        # 두 번째 subplot - 마스크
        mappable = ax[1].imshow(mask_np, cmap='jet')
        ax[1].set_title(f"Mask {i}")
        ax[1].axis('off')

        # colorbar
        fig.colorbar(mappable, ax=ax[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

visualize_samples(train_dataloader, n=2)

# --------------------------------------------------------------------------------
# 5) 모델(ViT) & 학습
# --------------------------------------------------------------------------------
# 예시: DeepLabV3Plus + ViT 인코더
model = smp.Segformer(
    encoder_name="mit_b5",         # MiT-B5 encoder (ViT 기반 계층형 백본)
    encoder_weights=None,    # 제공된 사전학습 가중치
    in_channels=3,
    classes=4
)
def pixel_accuracy(output, mask):
    with torch.no_grad():
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = (pred == mask).float().sum()
        total   = mask.numel()
        return float(correct / total)

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

        # Validation
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

        # Early Stopping
        if val_loss < min_val_loss:
            print(f"Val Loss improved {min_val_loss:.3f} -> {val_loss:.3f}. Saving model...")
            min_val_loss = val_loss
            no_improve   = 0
            torch.save(model.state_dict(), f"Unet_vit_epoch{e+1}_val{val_loss:.3f}.pth")
        else:
            no_improve += 1
            print(f"Val Loss not improved count = {no_improve}")
            if no_improve >= 7:
                print("No improvement for 7 epochs, stopping early.")
                break

# --------------------------------------------------------------------------------
# 6) 학습 실행
# --------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs=10
max_lr=1e-3
weight_decay=1e-4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dataloader)
)

fit(epochs, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler)
