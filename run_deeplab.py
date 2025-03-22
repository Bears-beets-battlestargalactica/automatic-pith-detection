# run_deeplab.py

import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F

# Dataset loader with augmentation
class PithDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_paths = sorted(Path(image_dir).glob("*.jpg"))
        self.mask_paths = sorted(Path(mask_dir).glob("*.png"))
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        if self.augment:
            img = self.augment_transform(img)
            mask = TF.to_tensor(mask).squeeze(0)
            mask = TF.resize(mask, [img.shape[1], img.shape[2]])  # Resize mask to match
        else:
            img = self.base_transform(img)
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 127).astype(np.float32)
            mask = torch.tensor(mask)

        return img, mask.unsqueeze(0)

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

        return img, mask

# Loss plot
def plot_loss(train_loss, val_loss):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_loss, mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(y=val_loss, mode='lines+markers', name='Val Loss'))
    fig.update_layout(title='DeepLabV3 Loss Curve', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark')
    fig.write_html("outputs/deeplab_loss.html")

# Metric calculation
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

# Visualization
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

# Training loop
def train(model, train_loader, val_loader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_losses, val_losses = [], []
    model.train()

    for epoch in range(epochs):
        running_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)['out']
            output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        val_loss = evaluate(model, val_loader, device, return_loss=True)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

    plot_loss(train_losses, val_losses)

# Evaluation loop
@torch.no_grad()
def evaluate(model, loader, device, save_dir=None, return_loss=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        output = model(x)['out']
        output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
        loss = criterion(output, y)
        total_loss += loss.item()
        output = torch.sigmoid(output).cpu().numpy()
        y = y.cpu().numpy()

        if save_dir:
            save_prediction(x[0].cpu(), output[0][0], f"{save_dir}/sample_{idx}.png")

        all_preds.append(output[0][0])
        all_labels.append(y[0][0])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_preds, all_labels)
    print(f"Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1: {metrics[3]:.4f}, IoU: {metrics[4]:.4f}")
    return total_loss / len(loader) if return_loss else metrics

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS backend")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    model.to(device)

    train_set = PithDataset("datasets/full_dataset/train/images", "datasets/full_dataset/train/masks", augment=True)
    val_set   = PithDataset("datasets/full_dataset/val/images", "datasets/full_dataset/val/masks")
    test_set  = PithDataset("datasets/full_dataset/test/images", "datasets/full_dataset/test/masks")

    train_loader = DataLoader(train_set, batch_size=6, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=6, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)

    print("\n>> Training DeepLabV3")
    train(model, train_loader, val_loader, device, epochs=10)

    print("\n>> Evaluation on test set")
    evaluate(model, test_loader, device, save_dir="outputs/deeplab_test_vis")
