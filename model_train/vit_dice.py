# -----------------------------------------------------------
# SegFormer-B0 + DiceLoss only
# -----------------------------------------------------------

import os, random, cv2, numpy as np, matplotlib.pyplot as plt, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

# ── 설정 ──
IMAGE_DIR    = r'C:/Users/hojoo/Desktop/ims/dataset2/gan_frame'
MASK_DIR     = r'C:/Users/hojoo/Desktop/ims/dataset2/gan_mask'
IMAGE_SIZE   = 256
NUM_SAMPLES  = 2700
BATCH_SIZE   = 8
EPOCHS       = 40
LR           = 6e-5
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR  = 'results_segformer_dice'
PLOTS_DIR    = os.path.join(RESULTS_DIR, 'plots')
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.txt')
SEED         = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
print('Device :', DEVICE)

# ── Dataset ──
class CovidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files):
        self.img_dir, self.mask_dir, self.files = img_dir, mask_dir, files
        self.proc = SegformerImageProcessor(do_resize=True, size=IMAGE_SIZE)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img  = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, fname.replace('frame_','mask_')),
                          cv2.IMREAD_GRAYSCALE)
        pix = self.proc(images=img, return_tensors='pt')['pixel_values'][0]      # (3,H,W)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask.astype(np.float32)/255.).unsqueeze(0)        # (1,H,W)
        return pix, mask

# ── DiceLoss 단독 ──
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)
        dice = (2*inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

# ── metric ──
@torch.no_grad()
def calc_metrics(preds, masks, thr=0.5):
    preds = (preds > thr).float(); masks = (masks > .5).float()
    inter = (preds * masks).sum((1,2,3)); union = ((preds + masks) > 0).sum((1,2,3))
    miou  = ((inter+1e-7)/(union+1e-7)).mean().item()
    tp, fp = inter, (preds*(1-masks)).sum((1,2,3))
    fn = ((1-preds)*masks).sum((1,2,3))
    prec = tp/(tp+fp+1e-7); rec = tp/(tp+fn+1e-7)
    f1 = (2*prec*rec)/(prec+rec+1e-7)
    return f1.mean().item(), miou

def plot_history(hist):
    ep = range(1, len(hist['train'])+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.plot(ep, hist['train'], label='train'); plt.plot(ep, hist['val'], label='val'); plt.title('Dice Loss'); plt.legend()
    plt.subplot(1,3,2); plt.plot(ep, hist['f1']); plt.title('Val F1')
    plt.subplot(1,3,3); plt.plot(ep, hist['miou']); plt.title('Val mIoU')
    plt.tight_layout(); pth = os.path.join(PLOTS_DIR,'history.png'); plt.savefig(pth); plt.close(); print('Saved:', pth)

# ── 데이터 분할 ──
files = sorted(os.listdir(IMAGE_DIR))[:NUM_SAMPLES]
trv, test = train_test_split(files, test_size=.2, random_state=SEED)
train_f, val_f = train_test_split(trv, test_size=.25, random_state=SEED)
train_loader = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, train_f), BATCH_SIZE, True,  pin_memory=True)
val_loader   = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, val_f),   BATCH_SIZE, False, pin_memory=True)
test_loader  = DataLoader(CovidDataset(IMAGE_DIR, MASK_DIR, test),    BATCH_SIZE, False, pin_memory=True)

# ── 모델 & 옵티마이저 ──
model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b0-finetuned-ade-512-512',
    ignore_mismatched_sizes=True,
    num_labels=1,
    id2label={0:'fg'}).to(DEVICE)
criterion = DiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda')

hist = {'train':[], 'val':[], 'f1':[], 'miou':[]}; best_iou = 0

# ── 학습 ──
for epoch in range(1, EPOCHS+1):
    model.train(); tr_loss = 0
    for x, y in tqdm(train_loader, desc=f'Train {epoch}/{EPOCHS}'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):
            logits = model(pixel_values=x).logits
            logits = torch.nn.functional.interpolate(logits, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
            loss = criterion(logits, y)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        tr_loss += loss.item()*x.size(0)
    tr_loss /= len(train_loader.dataset)

    model.eval(); val_loss=0; f1s=[]; mious=[]
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(pixel_values=x).logits
            logits = torch.nn.functional.interpolate(logits, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
            val_loss += criterion(logits, y).item()*x.size(0)
            f1, miou = calc_metrics(torch.sigmoid(logits), y); f1s.append(f1); mious.append(miou)
    val_loss/=len(val_loader.dataset); f1=np.mean(f1s); miou=np.mean(mious)

    hist['train'].append(tr_loss); hist['val'].append(val_loss); hist['f1'].append(f1); hist['miou'].append(miou)
    print(f'Epoch {epoch}/{EPOCHS} | Tr {tr_loss:.4f} | Val {val_loss:.4f} | F1 {f1:.4f} | mIoU {miou:.4f}')

    if miou > best_iou:
        best_iou = miou
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR,'segformer_best_dice.pt'))
        print('  ↳ best model saved')

# ── 학습 그래프 ──
plot_history(hist)

# ── 테스트 ──
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR,'segformer_best_dice.pt')))
model.eval(); test_loss=0; f1s=[]; mious=[]
with torch.no_grad():
    for x,y in tqdm(test_loader, desc='Test'):
        x,y = x.to(DEVICE), y.to(DEVICE)
        logits = model(pixel_values=x).logits
        logits = torch.nn.functional.interpolate(logits, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
        test_loss += criterion(logits, y).item()*x.size(0)
        f1,miou = calc_metrics(torch.sigmoid(logits), y); f1s.append(f1); mious.append(miou)
test_loss/=len(test_loader.dataset); test_f1=np.mean(f1s); test_miou=np.mean(mious)

with open(METRICS_FILE,'w') as f:
    f.write('=== Test Results ===\n')
    f.write(f'Test Loss : {test_loss:.4f}\nTest F1   : {test_f1:.4f}\nTest mIoU : {test_miou:.4f}\n')
print(f'Test Loss {test_loss:.4f} | F1 {test_f1:.4f} | mIoU {test_miou:.4f}')
print('Metrics saved ->', METRICS_FILE)
