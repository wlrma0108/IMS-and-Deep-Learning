import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 설정 ---
IMAGE_DIR    = r'C:\Users\hojoo\Desktop\ims\dataset2_aug\dataset2_elastic'
MASK_DIR     = r'C:\Users\hojoo\Desktop\ims\dataset2_aug\dataset2_elastic_masks'
IMAGE_SIZE   = 256
NUM_SAMPLES  = 2700
BATCH_SIZE   = 8
EPOCHS       = 100
LEARNING_RATE= 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR  = 'results'
PLOTS_DIR    = os.path.join(RESULTS_DIR, 'plots')
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.txt')

print(f"사용중인 장치 : {DEVICE}")

# --- 결과 폴더 생성 ---
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Dataset 클래스 ---
class CovidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, samples):
        self.files = samples
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)

        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # (1,H,W)

        return torch.tensor(img), torch.tensor(mask)

# --- UNet 모델 ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bridge = ConvBlock(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x); p1 = self.pool1(s1)
        s2 = self.enc2(p1); p2 = self.pool2(s2)
        s3 = self.enc3(p2); p3 = self.pool3(s3)
        s4 = self.enc4(p3); p4 = self.pool4(s4)
        b  = self.bridge(p4)
        d1 = self.up1(b);   d1 = torch.cat([d1, s4], dim=1); d1 = self.dec1(d1)
        d2 = self.up2(d1);  d2 = torch.cat([d2, s3], dim=1); d2 = self.dec2(d2)
        d3 = self.up3(d2);  d3 = torch.cat([d3, s2], dim=1); d3 = self.dec3(d3)
        d4 = self.up4(d3);  d4 = torch.cat([d4, s1], dim=1); d4 = self.dec4(d4)
        return torch.sigmoid(self.out(d4))

# --- Metric 계산 함수 ---
def calculate_metrics(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    masks = (masks > 0.5).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union        = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
    iou          = (intersection + 1e-7) / (union + 1e-7)
    miou         = iou.mean().item()

    tp    = intersection
    fp    = (preds * (1 - masks)).sum(dim=(1, 2, 3))
    fn    = ((1 - preds) * masks).sum(dim=(1, 2, 3))
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = (2 * precision * recall) / (precision + recall + 1e-7)
    f1_mean   = f1.mean().item()

    return f1_mean, miou

# --- Train & Eval ---
def train_model(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    f1_scores, ious = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss  = criterion(preds, masks)
            running_loss += loss.item() * imgs.size(0)
            f1, miou = calculate_metrics(preds, masks)
            f1_scores.append(f1); ious.append(miou)
    return running_loss / len(loader.dataset), np.mean(f1_scores), np.mean(ious)

# --- 시각화 및 저장 ---
def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.title('Loss'); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_f1'],     label='Val F1')
    plt.title('F1 Score'); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_miou'],   label='Val mIoU')
    plt.title('mIoU'); plt.legend()

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'training_history.png')
    plt.savefig(save_path)
    print(f'학습 곡선이 저장되었습니다: {save_path}')
    plt.close()

# --- 메인 실행 ---
if __name__ == '__main__':
    # 데이터 분할
    files     = os.listdir(IMAGE_DIR)[:NUM_SAMPLES]
    train_val, test = train_test_split(files, test_size=0.2, random_state=42)
    train, val      = train_test_split(train_val, test_size=0.25, random_state=42)

    # Dataset & DataLoader
    train_loader = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, test),  batch_size=BATCH_SIZE)

    # 모델/옵티마이저/손실함수
    model     = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 루프
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_miou': []}
    for epoch in range(1, EPOCHS + 1):
        train_loss                = train_model(model, train_loader, criterion, optimizer)
        val_loss, val_f1, val_miou = eval_model(model, val_loader,   criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_miou'].append(val_miou)

        print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | mIoU: {val_miou:.4f}')

    # 모델 저장
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'ela_unet_covid_segmentation.pt'))

    # 학습 곡선 저장
    plot_history(history)

    # 테스트 데이터 평가
    test_loss, test_f1, test_miou = eval_model(model, test_loader, criterion)
    metrics_text = (
        f"Test Loss : {test_loss:.4f}\n"
        f"Test F1   : {test_f1:.4f}\n"
        f"Test mIoU : {test_miou:.4f}\n"
    )
    with open(METRICS_FILE, 'w') as f:
        f.write("=== Test Results ===\n")
        f.write(metrics_text)
    print(metrics_text)
    print(f'테스트 결과가 저장되었습니다: {METRICS_FILE}')
    