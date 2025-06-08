import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configuration
IMAGE_DIR = r'C:\Users\hojoo\Downloads\dataset2\frames'
MASK_DIR  = r'C:/Users/hojoo/Downloads/dataset2/masks'
IMAGE_SIZE = 512
NUM_SAMPLES = 40  # 필요에 맞게 조정
BATCH_SIZE = 2
EPOCHS = 100
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_image(path, size=IMAGE_SIZE, color=True):
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size))
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


class CovidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, samples):
        self.files = samples
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # --- 이미지 읽기 (RGB 3채널) ---
        img = cv2.imread(os.path.join(self.image_dir, fname), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)

        # --- 마스크 읽기 (GRAY 1채널) ---
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # (1,H,W)

        return torch.tensor(img), torch.tensor(mask)



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
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        b = self.bridge(p4)

        d1 = self.up1(b)
        d1 = torch.cat([d1, s4], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, s3], dim=1)
        d2 = self.dec2(d2)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, s2], dim=1)
        d3 = self.dec3(d3)
        d4 = self.up4(d3)
        d4 = torch.cat([d4, s1], dim=1)
        d4 = self.dec4(d4)

        return torch.sigmoid(self.out(d4))


def train_model(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


if __name__ == '__main__':
    files = os.listdir(IMAGE_DIR)[:NUM_SAMPLES]
    X_train, X_test = train_test_split(files, test_size=1 - VALIDATION_SPLIT, random_state=42)

    train_ds = CovidDataset(IMAGE_DIR, MASK_DIR, X_train)
    test_ds  = CovidDataset(IMAGE_DIR, MASK_DIR, X_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS+1):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss   = eval_model(model, test_loader, criterion)
        print(f'Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save
    torch.save(model.state_dict(), 'unet_covid_segmentation.pt')