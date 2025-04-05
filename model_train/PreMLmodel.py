import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------------------
# (선택) 세그멘테이션용 .npy 파일 로드 (현재 분류 코드에서는 사용 안 함)
# ----------------------------------------------------------------------------------
DATASET_DIR = "./dataset"  # .npy 파일들이 들어있는 폴더
images_radiopedia = np.load(os.path.join(DATASET_DIR, "images_radiopedia.npy"))
masks_radiopedia  = np.load(os.path.join(DATASET_DIR, "masks_radiopedia.npy"))
images_medseg     = np.load(os.path.join(DATASET_DIR, "images_medseg.npy"))
masks_medseg      = np.load(os.path.join(DATASET_DIR, "masks_medseg.npy"))
test_images_medseg= np.load(os.path.join(DATASET_DIR, "test_images_medseg.npy"))
# 위 배열들은 세그멘테이션 등에 활용 가능 (현재 예제는 분류 모델이므로 미사용)

# ----------------------------------------------------------------------------------
# 분류용 설정
# ----------------------------------------------------------------------------------
DATASET_ROOT = "./dataset"  # ImageFolder 경로 (하위에 4개 클래스 폴더)
BATCH_SIZE   = 16
NUM_EPOCHS   = 5
LEARNING_RATE= 1e-4
NUM_CLASSES  = 4  # 분류할 클래스 수
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------
# 1. 데이터 전처리 & Dataloader 구성
# ----------------------------------------------------------------------------------
def prepare_dataloaders(dataset_root, batch_size=16, split_ratio=0.8):
    """
    ImageFolder 구조의 폴더(dataset_root)를 불러와서
    학습/검증 세트로 분할 후 DataLoader 반환
    """
    # (1) Transform 정의
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # (2) 전체 ImageFolder 로드
    full_dataset = datasets.ImageFolder(root=dataset_root)
    class_names  = full_dataset.classes  # 예: ['class0', 'class1', 'class2', 'class3']

    # (3) 학습/검증 분할
    dataset_size = len(full_dataset)
    train_size   = int(dataset_size * split_ratio)
    val_size     = dataset_size - train_size

    indices      = torch.randperm(dataset_size).tolist()
    train_indices= indices[:train_size]
    val_indices  = indices[train_size:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset   = Subset(full_dataset, val_indices)

    # (4) 각각 Transform 적용
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform   = val_transform

    # (5) DataLoader 구성
    train_loader= DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader  = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names


# ----------------------------------------------------------------------------------
# 2. 모델 생성 함수 (VGG, DenseNet, ConvNeXt, VisionTransformer)
# ----------------------------------------------------------------------------------
def create_vgg(num_classes, pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def create_densenet(num_classes, pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, num_classes)
    return model

def create_convnext(num_classes, pretrained=True):
    model = models.convnext_tiny(pretrained=pretrained)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def create_vit(num_classes, pretrained=True):
    """
    VisionTransformer (vit_b_16) 은 torchvision >= 0.13 에서만 사용 가능
    """
    model = models.vit_b_16(pretrained=pretrained)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ----------------------------------------------------------------------------------
# 3. 학습 및 평가 함수
# ----------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    한 epoch 학습 & (avg_loss, accuracy) 반환
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, class_names):
    """
    검증 모드: (avg_loss, accuracy, classification_report, confusion_matrix)
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   zero_division=0)
    cm     = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, report, cm


# ----------------------------------------------------------------------------------
# 4. 메인(여러 모델 학습 & 성능 비교)
# ----------------------------------------------------------------------------------
def main():
    # (1) DataLoader 준비
    train_loader, val_loader, class_names = prepare_dataloaders(
        dataset_root=DATASET_ROOT, 
        batch_size=BATCH_SIZE, 
        split_ratio=0.8
    )
    print(f"[INFO] Classes: {class_names}")

    # (2) 모델 팩토리 (이름→생성 함수)
    model_factories = {
        "vgg16": create_vgg,
        "densenet121": create_densenet,
        "convnext_tiny": create_convnext,
        "vit_b_16": create_vit,
    }

    criterion = nn.CrossEntropyLoss()
    results   = {}

    # (3) 모델별 학습
    for model_name, factory_func in model_factories.items():
        print(f"\n=== Training {model_name} ===")
        model = factory_func(num_classes=NUM_CLASSES, pretrained=True).to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_acc     = 0.0
        best_weights = None

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_report, val_cm = evaluate(model, val_loader, criterion, device, class_names)

            print(f"[{epoch+1}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 최고 정확도 갱신 시 가중치 저장
            if val_acc > best_acc:
                best_acc     = val_acc
                best_weights = model.state_dict()

        # 최고 성능 모델 복원
        model.load_state_dict(best_weights)

        # 최종 결과
        final_loss, final_acc, final_report, final_cm = evaluate(model, val_loader, criterion, device, class_names)
        print(f"\n>>> {model_name} Best Val Accuracy: {best_acc:.4f}")
        print("Classification Report:")
        print(final_report)
        print("Confusion Matrix:")
        print(final_cm)

        results[model_name] = best_acc

    # (4) 모델별 결과 비교
    print("\n=== Model Comparison ===")
    for m, acc in results.items():
        print(f"{m}: {acc:.4f}")

    best_model_name = max(results, key=results.get)
    print(f"\n>>> Best model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")


if __name__ == "__main__":
    main()
