# run_unet.py

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------
# UNet Model
# ------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = conv_block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        b = self.bottleneck(self.pool4(d4))
        u1 = self.dec1(torch.cat([self.up1(b), d4], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u1), d3], dim=1))
        u3 = self.dec3(torch.cat([self.up3(u2), d2], dim=1))
        u4 = self.dec4(torch.cat([self.up4(u3), d1], dim=1))
        return self.sigmoid(self.final(u4))

# ------------------------------
# Dataset with Augmentation
# ------------------------------
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
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(1.0, 1.0))
            img = TF.resized_crop(img, i, j, h, w, (256, 256))
            mask = TF.resized_crop(mask, i, j, h, w, (256, 256))

        img = TF.to_tensor(img)
        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) / 255.

        return img, mask

# ------------------------------
# Metric Calculation
# ------------------------------
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

# ------------------------------
# Save Output Image with Mask
# ------------------------------
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

# ------------------------------
# Training Loop
# ------------------------------
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
        train_losses.append(total_loss / len(loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # Plot loss curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, name='Train Loss'))
    fig.add_trace(go.Scatter(y=val_losses, name='Val Loss'))
    fig.update_layout(title='U-Net Training & Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
    os.makedirs("outputs", exist_ok=True)
    fig.write_html("outputs/unet_loss.html")

# ------------------------------
# Evaluation Loop
# ------------------------------
def evaluate(model, loader, device, save_dir=None, verbose=False):
    model.eval()
    all_preds, all_labels = [], []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            pred = model(x).cpu().numpy()
            label = y.cpu().numpy()
            if save_dir:
                save_prediction(x[0].cpu(), pred[0][0], f"{save_dir}/sample_{idx}.png")
            all_preds.append(pred[0][0])
            all_labels.append(label[0][0])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_preds, all_labels)
    if verbose:
        print(f"Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, "
              f"F1: {metrics[3]:.4f}, IoU: {metrics[4]:.4f}")
    return metrics

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = UNet().to(device)

    train_data = PithDataset("datasets/full_dataset/train/images", "datasets/full_dataset/train/masks", augment=True)
    val_data   = PithDataset("datasets/full_dataset/val/images", "datasets/full_dataset/val/masks")
    test_data  = PithDataset("datasets/full_dataset/test/images", "datasets/full_dataset/test/masks")

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print("\n>> Training U-Net")
    train(model, train_loader, val_loader, device, epochs=10)

    print("\n>> Evaluation on full test set")
    evaluate(model, test_loader, device, save_dir="outputs/unet_test_vis", verbose=True)
