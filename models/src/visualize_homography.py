"""
Visualize predicted vs. ground truth homographies on sample images.
Draws the warped image corners for both predicted and true homographies.
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.src.homography_dataset import HomographyDataset
from models.src.homography_regressor import HomographyRegressor
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Config ---
IMAGES_DIR = 'data/e2e_frames'
CALIBRATION_DIR = 'data/calibration'
MAPPING_FILE = 'data/homography_mapping.csv'
MODEL_PATH = 'checkpoints/homography_model_epoch20.pth'  # Change as needed
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 10  # Number of samples to visualize

# --- Load mapping (CSV) ---
mapping = []
with open(MAPPING_FILE, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) >= 2:
            mapping.append([row[0], row[1]])

# Use last 20% as validation
val_mapping = mapping[int(0.8*len(mapping)):]  # last 20%

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = HomographyDataset(IMAGES_DIR, CALIBRATION_DIR, val_mapping, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = HomographyRegressor(backbone='resnet18', pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

corners = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
], dtype=np.float32).T  # shape (3, 4)

unnormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def plot_sample(image_tensor, H_gt, H_pred, idx):
    # Unnormalize and convert to numpy
    image = unnormalize(image_tensor.squeeze(0)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    H_gt_np = H_gt.cpu().numpy()
    H_pred_np = H_pred.cpu().numpy()
    # Project corners
    gt_proj = H_gt_np @ corners
    pred_proj = H_pred_np @ corners
    gt_proj = gt_proj[:2] / gt_proj[2]
    pred_proj = pred_proj[:2] / pred_proj[2]
    # Scale to image size
    h, w = image.shape[:2]
    wh = np.array([[w], [h]])  # shape (2,1)
    gt_proj = gt_proj * wh
    pred_proj = pred_proj * wh
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    gt_closed = np.hstack([gt_proj, gt_proj[:, 0:1]])
    pred_closed = np.hstack([pred_proj, pred_proj[:, 0:1]])
    plt.plot(gt_closed[0], gt_closed[1], 'g-', label='GT')
    plt.plot(pred_closed[0], pred_closed[1], 'r--', label='Pred')
    plt.scatter(gt_proj[0], gt_proj[1], c='g')
    plt.scatter(pred_proj[0], pred_proj[1], c='r')
    plt.legend(['GT corners', 'Pred corners'])
    plt.title(f'Sample {idx}: Green=GT, Red=Pred')
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('outputs/homography_viz', exist_ok=True)
    plt.savefig(f'outputs/homography_viz/sample_{idx}.png')
    plt.close()

with torch.no_grad():
    for idx, (images, H_gt) in enumerate(dataloader):
        if idx >= N_SAMPLES:
            break
        images = images.to(DEVICE)
        H_gt = H_gt.to(DEVICE)
        H_pred = model(images)
        plot_sample(images, H_gt[0], H_pred[0], idx)
