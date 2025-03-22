# run_swin.py

import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timm

# Swin Transformer Segmentation Model
class SwinTransformerSegmentation(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            features_only=True
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.backbone.feature_info[-1]['num_chs'], 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.backbone(x)
        x = self.decoder(feats[-1])
        return x

# Dataset
class PithDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
        self.mask_paths = sorted(Path(mask_dir).glob("*.png"))
        self.augment = augment
        self.jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.crop = transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32)

        img = TF.to_pil_image(img)
        mask = TF.to_pil_image(mask)

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            angle = random.uniform(-30, 30)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
            img = self.jitter(img)
            img = self.crop(img)
            mask = self.crop(mask)

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)

        return img, mask

# Metrics
def calculate_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    pred_bin = (pred > 0.5).astype(np.uint8)
    acc = accuracy_score(target, pred_bin)
    prec = precision_score(target, pred_bin, zero_division=0)
    rec = recall_score(target, pred_bin, zero_division=0)
    f1 = f1_score(target, pred_bin, zero_division=0)

    intersection = np.logical_and(pred_bin, target).sum()
    union = np.logical_or(pred_bin, target).sum()
    iou = intersection / (union + 1e-6)

    return acc, prec, rec, f1, iou

# Visualize prediction
def save_prediction(img_tensor, pred_mask, out_path):
    img = img_tensor.permute(1, 2, 0).numpy()
    pred = (pred_mask > 0.5).astype(np.uint8) * 255

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(pred, alpha=0.4, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Optional NMS (mask-wise thresholding)
def apply_nms_mask(preds, threshold=0.5):
    return np.where(preds > threshold, preds, 0)

# Training
def train(model, loader, val_loader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(loader)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # Plot loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, name='Train Loss'))
    fig.add_trace(go.Scatter(y=val_losses, name='Val Loss'))
    fig.update_layout(title='Training & Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
    os.makedirs("outputs", exist_ok=True)
    fig.write_html("outputs/swin_loss.html")

# Evaluation
def evaluate(model, loader, device, nms=False, save_dir=None, verbose=False):
    model.eval()
    all_preds, all_labels = [], []

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            pred = model(x).cpu().numpy()
            label = y.cpu().numpy()

            if nms:
                pred = apply_nms_mask(pred)

            if save_dir:
                save_prediction(x[0].cpu(), pred[0][0], f"{save_dir}/sample_{idx}.png")

            all_preds.append(pred[0][0])
            all_labels.append(label[0][0])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_preds, all_labels)

    if verbose:
        print(f"Accuracy: {metrics[0]:.4f} | Precision: {metrics[1]:.4f} | Recall: {metrics[2]:.4f} | F1: {metrics[3]:.4f} | IoU: {metrics[4]:.4f}")
    return metrics

# Main
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SwinTransformerSegmentation(pretrained=True).to(device)

    train_data = PithDataset("datasets/full_dataset/train/images", "datasets/full_dataset/train/masks", augment=True)
    val_data   = PithDataset("datasets/full_dataset/val/images", "datasets/full_dataset/val/masks")
    test_data  = PithDataset("datasets/full_dataset/test/images", "datasets/full_dataset/test/masks")
    oak_data   = PithDataset("datasets/oak_dataset/images", "datasets/oak_dataset/masks")

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    oak_loader = DataLoader(oak_data, batch_size=1, shuffle=False)

    print("\n>> Training Swin Transformer")
    train(model, train_loader, val_loader, device, epochs=10)

    print("\n>> Evaluation on test set")
    evaluate(model, test_loader, device, save_dir="outputs/swin_test_vis", verbose=True)

    print("\n>> Evaluation on oak dataset (NO NMS)")
    evaluate(model, oak_loader, device, save_dir="outputs/swin_oak_raw", nms=False, verbose=True)

    print("\n>> Evaluation on oak dataset (WITH NMS)")
    evaluate(model, oak_loader, device, save_dir="outputs/swin_oak_nms", nms=True, verbose=True)
