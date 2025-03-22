# run_yolov9.py
"""

# configs/pith.yaml
train: datasets/full_dataset/train/images
val: datasets/full_dataset/val/images

nc: 1
names: ['pith']

# Augmentation parameters used during training
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 30.0
translate: 0.1
scale: 0.5
shear: 10.0
flipud: 0.5
fliplr: 0.5
"""

import os
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def train_yolo(model_name='yolov9c.pt', data_yaml='configs/pith.yaml', epochs=25, imgsz=640):
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        lr0=1e-4,
        optimizer='Adam',
        project='runs',
        name='yolov9_pith',
        exist_ok=True
    )
    return model, results

def plot_loss_curve(log_file='runs/yolov9_pith/results.csv', output_html='outputs/yolov9_loss.html'):
    df = pd.read_csv(log_file)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['train/box_loss'], name='Train Box Loss'))
    fig.add_trace(go.Scatter(y=df['val/box_loss'], name='Val Box Loss'))
    fig.update_layout(title='YOLOv9 Training & Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
    os.makedirs("outputs", exist_ok=True)
    fig.write_html(output_html)

def evaluate_yolo(model_path, img_dir, label_dir, save_vis_dir):
    model = YOLO(model_path)
    os.makedirs(save_vis_dir, exist_ok=True)
    image_paths = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))

    ious, all_preds, all_gts = [], [], []

    for img_path in tqdm(image_paths, desc=f"Evaluating on {Path(img_dir).name}"):
        results = model(img_path, imgsz=640, conf=0.25)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        img = cv2.imread(str(img_path))

        # Save bounding box visualization
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(f"{save_vis_dir}/{img_path.stem}_pred.jpg", img)

        # Load GT boxes
        label_file = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_file.exists():
            continue

        gt_boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                _, cx, cy, w, h = map(float, line.strip().split())
                x1 = int((cx - w / 2) * img.shape[1])
                y1 = int((cy - h / 2) * img.shape[0])
                x2 = int((cx + w / 2) * img.shape[1])
                y2 = int((cy + h / 2) * img.shape[0])
                gt_boxes.append([x1, y1, x2, y2])

        # IoU calculation
        for pred_box in boxes:
            best_iou = 0
            for gt_box in gt_boxes:
                xi1 = max(pred_box[0], gt_box[0])
                yi1 = max(pred_box[1], gt_box[1])
                xi2 = min(pred_box[2], gt_box[2])
                yi2 = min(pred_box[3], gt_box[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                boxA = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                boxB = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                union_area = boxA + boxB - inter_area
                iou = inter_area / (union_area + 1e-6)
                best_iou = max(best_iou, iou)
            ious.append(best_iou)

        # Binary predictions (based on IoU > 0.5)
        for gt_box in gt_boxes:
            matched = any(
                ((min(pred_box[2], gt_box[2]) - max(pred_box[0], gt_box[0])) *
                 (min(pred_box[3], gt_box[3]) - max(pred_box[1], gt_box[1])) /
                 ((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]) +
                  (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) -
                  max(0, min(pred_box[2], gt_box[2]) - max(pred_box[0], gt_box[0])) *
                  max(0, min(pred_box[3], gt_box[3]) - max(pred_box[1], gt_box[1])) + 1e-6)) > 0.5
                for pred_box in boxes
            )
            all_preds.append(1 if matched else 0)
            all_gts.append(1)

        for _ in range(len(boxes) - len(gt_boxes)):
            all_preds.append(1)
            all_gts.append(0)

    if ious:
        avg_iou = np.mean(ious)
        precision = precision_score(all_gts, all_preds, zero_division=0)
        recall = recall_score(all_gts, all_preds, zero_division=0)
        f1 = f1_score(all_gts, all_preds, zero_division=0)

        print(f"\n📊 Evaluation Results for {Path(img_dir).name}")
        print(f"Mean IoU     : {avg_iou:.4f}")
        print(f"Precision    : {precision:.4f}")
        print(f"Recall       : {recall:.4f}")
        print(f"F1 Score     : {f1:.4f}")
    else:
        print(f"\n⚠️ No valid annotations found in {label_dir}")

if __name__ == "__main__":
    print("\n🚀 Training YOLOv9...")
    model, results = train_yolo()

    print("\n📈 Plotting training loss...")
    plot_loss_curve()

    print("\n🔍 Evaluating on test set...")
    evaluate_yolo(
        model_path='runs/yolov9_pith/weights/best.pt',
        img_dir='datasets/full_dataset/test/images',
        label_dir='datasets/full_dataset/test/labels',
        save_vis_dir='outputs/yolov9_test_vis'
    )

    print("\n🌳 Evaluating on oak dataset...")
    evaluate_yolo(
        model_path='runs/yolov9_pith/weights/best.pt',
        img_dir='datasets/oak_dataset/images',
        label_dir='datasets/oak_dataset/labels',
        save_vis_dir='outputs/yolov9_oak_vis'
    )
