
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.src.homography_dataset import HomographyDataset
from models.src.homography_regressor import HomographyRegressor
import csv

# --- Config ---
IMAGES_DIR = 'data/e2e_frames'
CALIBRATION_DIR = 'data/calibration'
MAPPING_FILE = 'data/homography_mapping.csv'  # CSV file with image_path,calibration_json_path
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # --- Data Preparation ---
    mapping = []
    with open(MAPPING_FILE, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                mapping.append([row[0], row[1]])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = HomographyDataset(IMAGES_DIR, CALIBRATION_DIR, mapping, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Model ---
    model = HomographyRegressor(backbone='resnet18', pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, H_gt in dataloader:
            images = images.to(DEVICE)
            H_gt = H_gt.to(DEVICE)
            optimizer.zero_grad()
            H_pred = model(images)
            loss = criterion(H_pred, H_gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f}")
        # Save checkpoint
        torch.save(model.state_dict(), f'checkpoints/homography_model_epoch{epoch+1}.pth')

    print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()
