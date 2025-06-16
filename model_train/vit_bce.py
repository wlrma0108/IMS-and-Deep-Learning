"""SegFormer‑B0 Semantic Segmentation (BCE‑only Loss)
기존 파이프라인에서 DiceLoss를 제거하고 **BCEWithLogitsLoss 하나만** 사용합니다.
"""
import os, random, cv2, numpy as np, matplotlib.pyplot as plt, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

# ───────────────────────── 설정 ──────────────────────────
IMAGE_DIR    = r'C:/Users/hojoo/Desktop/ims/dataset2/gan_frame'
MASK_DIR     = r'C:/Users/hojoo/Desktop/ims/dataset2/gan_mask'
IMAGE_SIZE   = 256
NUM_SAMPLES  = 2700
BATCH_SIZE   = 8
EPOCHS       = 40
LEARNING_RATE= 6e-5
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR  = 'results_segformer'; PLOTS_DIR = os.path.join(RESULTS_DIR,'plots')
METRICS_FILE = os.path.join(RESULTS_DIR,'metrics.txt')
SEED         = 42
os.makedirs(PLOTS_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)
print('Device:', DEVICE)

# ─────────────────────── Dataset ────────────────────────
class CovidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files):
        self.img_dir, self.mask_dir, self.files = img_dir, mask_dir, files
        self.proc = SegformerImageProcessor(do_resize=True, size=IMAGE_SIZE)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname)); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, fname.replace('frame_', 'mask_')), cv2.IMREAD_GRAYSCALE)
        pix = self.proc(images=img, return_tensors='pt')['pixel_values'][0]
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        return pix, mask

# ─────────────────────── Metrics ─────────────────────────
@torch.no_grad()
def calc_metrics(preds, masks, thr=0.5):
    preds = (preds > thr).float(); masks = (masks > 0.5).float()
    inter = (preds * masks).sum((1, 2, 3)); union = ((preds + masks) > 0).float().sum((1, 2, 3))
    miou = ((inter + 1e-7) / (union + 1e-7)).mean().item()
    tp = inter; fp = (preds * (1 - masks)).sum((1, 2, 3)); fn = ((1 - preds) * masks).sum((1, 2, 3))
    prec = tp / (tp + fp + 1e-7); rec = tp / (tp + fn + 1e-7)
    f1 = (2 * prec * rec) / (prec + rec + 1e-7)
    return f1.mean().item(), miou

def plot_history(hist):
    epochs = range(1, len(hist['train_loss']) + 1)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1); plt.plot(epochs, hist['train_loss'], label='Train'); plt.plot(epochs, hist['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1, 3, 2); plt.plot(epochs, hist['val_f1']); plt.title('Val F1')
    plt.subplot(1, 3, 3); plt.plot(epochs, hist['val_miou']); plt.title('Val mIoU')
    plt.tight_layout(); path = os.path.join(PLOTS_DIR, 'history.png'); plt.savefig(path); plt.close(); print('Saved plot ->', path)

# ─────────────────────── Data Split ──────────────────────
files = sorted(os.listdir(IMAGE_DIR))[:NUM_SAMPLES]
train_val, test = train_test_split(files, test_size=0.2, random_state=SEED)
train_files, val_files = train_test_split(train_val, test_size=0.25, random_state=SEED)
train_loader = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, train_files), BATCH_SIZE, True, pin_memory=True)
val_loader   = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, val_files),   BATCH_SIZE, False, pin_memory=True)
test_loader  = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, test),        BATCH_SIZE, False, pin_memory=True)

# ─────────────────────── Model & Opt ─────────────────────
model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512', ignore_mismatched_sizes=True, num_labels=1, id2label={0: 'fg'}).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')

history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_miou': []}
best_iou = 0

for epoch in range(1, EPOCHS + 1):
    # ----- Train -----
    model.train(); train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f'Train {epoch}/{EPOCHS}'):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
            logits = model(pixel_values=imgs).logits
            logits = torch.nn.functional.interpolate(logits, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
            loss = criterion(logits, masks)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # ----- Validation -----
    model.eval(); val_loss = 0; f1s = []; mious = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(pixel_values=imgs).logits
            logits = torch.nn.functional.interpolate(logits, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
            val_loss += criterion(logits, masks).item() * imgs.size(0)
            f1, miou = calc_metrics(torch.sigmoid(logits), masks); f1s.append(f1); mious.append(miou)
    val_loss /= len(val_loader.dataset); val_f1 = np.mean(f1s); val_miou = np.mean(mious)

    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1); history['val_miou'].append(val_miou)

    print(f'Epoch {epoch}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f} | F1 {val_f1:.4f} | mIoU {val_miou:.4f}')
    if val_miou > best_iou:
        best_iou = val_miou; torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'segformer_best.pt'))
        print('✓ best model saved')

# ─────────────────────── Plot & Test ─────────────────────
plot_history(history)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'segformer_best.pt')))
model.eval(); test_loss = 0; f1s = []; mious = []
with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc='Test'):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(pixel_values=imgs).logits
        logits = torch.nn.functional.interpolate(logits, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
        test_loss += criterion(logits, masks).item() * imgs.size(0)
        f1, miou = calc_metrics(torch.sigmoid(logits), masks); f1s.append(f1); mious.append(miou)

test_loss /= len(test_loader.dataset); test_f1 = np.mean(f1s); test_miou = np.mean(mious)
with open(METRICS_FILE, 'w') as f:
    f.write('=== Test Results ===\n'); f.write(f'Test Loss : {test_loss:.4f}\nTest F1   : {test_f1:.4f}\nTest mIoU : {test_miou:.4f}\n')
print(f'Test Loss {test_loss:.4f} | F1 {test_f1:.4f} | mIoU {test_miou:.4f}')
print(f'Test results saved to {METRICS_FILE}')