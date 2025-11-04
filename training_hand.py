print("Installing dependencies...")
!pip install albumentations opencv-python -q

print("Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class AdvancedHandDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None, augment_ai=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment_ai = augment_ai

        self.heavy_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.GaussNoise(p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.6),
            A.CLAHE(clip_limit=4.0, p=0.4),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.4),
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-20, 20),
                p=0.6
            ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.augment_ai and label == 0:
            image_np = np.array(image)
            image_np = self.heavy_aug(image=image_np)['image']
            image = Image.fromarray(image_np)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def load_dataset_with_stats(ai_folder, real_folder):
    image_paths = []
    labels = []

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    ai_path = Path(ai_folder)
    ai_count = 0
    if ai_path.exists():
        for img_file in ai_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(img_file))
                labels.append(0)
                ai_count += 1

    real_path = Path(real_folder)
    real_count = 0
    if real_path.exists():
        for img_file in real_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(img_file))
                labels.append(1)
                real_count += 1

    print(f"\nDataset Statistics:")
    print(f"  AI images: {ai_count}")
    print(f"  Real images: {real_count}")
    print(f"  Total: {ai_count + real_count}")
    print(f"  Class ratio (AI/Real): {ai_count / max(real_count, 1):.2f}")

    if ai_count < 200 or real_count < 200:
        print("\n⚠️  WARNING: Small dataset detected!")
        print("   - Will use aggressive augmentation")
        print("   - Will use class weighting")
        print("   - Consider collecting more data for better results")

    if abs(ai_count - real_count) > 100:
        print("\n⚠️  WARNING: Imbalanced dataset!")
        print("   - Will use weighted sampling to balance classes")

    return image_paths, labels, ai_count, real_count

def get_balanced_sampler(labels):
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def get_advanced_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def create_optimized_model(pretrained=True):
    if pretrained:
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_b0(weights=None)

    for i, (name, param) in enumerate(model.features.named_parameters()):
        if i < 100:
            param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )

    return model


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                              scheduler, device, epochs=100, patience=15):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc='Training'):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)
        print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ New best model! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Loaded best model (Val Acc: {best_val_acc:.2f}%)")

    return model, train_losses, val_losses, train_accs, val_accs

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(val_accs, label='Val Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    save_path = '/content/drive/MyDrive/hand_dataset/training_curves.png'
    plt.savefig(save_path)
    plt.close()
    print(f"\n✓ Training curves saved to {save_path}")


def main():
    AI_FOLDER = '/content/drive/MyDrive/hand_dataset/ai_hands'
    REAL_FOLDER = '/content/drive/MyDrive/hand_dataset/real_hands'
    MODEL_SAVE_PATH = "/content/drive/MyDrive/hand_dataset/hand_classifier_colab.pth"

    BATCH_SIZE = 16
    LEARNING_RATE = 0.0003
    EPOCHS = 100
    PATIENCE = 15

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_paths, labels, ai_count, real_count = load_dataset_with_stats(AI_FOLDER, REAL_FOLDER)
    if len(image_paths) < 100:
        print("\n✗ ERROR: Not enough images for training! You need at least 100 total images.")
        return

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nDataset Split: Training: {len(train_paths)} | Validation: {len(val_paths)}")

    train_transform, val_transform = get_advanced_transforms()
    train_dataset = AdvancedHandDataset(train_paths, train_labels, train_transform, augment_ai=(ai_count < real_count))
    val_dataset = AdvancedHandDataset(val_paths, val_labels, val_transform)
    train_sampler = get_balanced_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = create_optimized_model(pretrained=True).to(device)
    pos_weight = torch.tensor([ai_count / real_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model, train_losses, val_losses, train_accs, val_accs = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs=EPOCHS, patience=PATIENCE
    )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    if max(val_accs) < 70:
        print("\n⚠️  WARNING: Low accuracy! Consider collecting more diverse data.")

if __name__ == "__main__":
    main()