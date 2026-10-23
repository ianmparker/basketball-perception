import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset

class BallCourtE2EDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        # Always use absolute path to data/e2e_frames
        self.img_dir = os.path.abspath(os.path.join(os.getcwd(), 'data', 'e2e_frames'))
        self.transform = transform
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Each row: image, pixel_x, pixel_y, court_x, court_y
                self.samples.append({
                    'image': row['image'],
                    'pixel_x': float(row['pixel_x']),
                    'pixel_y': float(row['pixel_y']),
                    'court_x': float(row['court_x']),
                    'court_y': float(row['court_y'])
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Use the full relative path from the CSV, joined to project root if needed
        img_path = os.path.join(self.img_dir, os.path.basename(sample['image']))
        print(f"[DEBUG] Loading image: {img_path}")
        print(f"[DEBUG] Loading image: {img_path}")
        print(f"[DEBUG] Loading image: {img_path}")
        print(f"[DEBUG] Loading image: {os.path.abspath(img_path)}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Input: image, [pixel_x, pixel_y]
        pixel_coords = torch.tensor([sample['pixel_x'], sample['pixel_y']], dtype=torch.float32)
        # Target: [court_x, court_y]
        court_coords = torch.tensor([sample['court_x'], sample['court_y']], dtype=torch.float32)
        return image, pixel_coords, court_coords

# Example usage:
# from torchvision import transforms
# dataset = BallCourtE2EDataset('data/e2e_training_data.csv', 'data/e2e_frames/', transform=transforms.ToTensor())
# image, pixel_coords, court_coords = dataset[0]
