import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class HomographyDataset(Dataset):
    """
    Dataset for (image, homography matrix) pairs.
    Expects:
        - images_dir: directory with broadcast frames
        - calibration_dir: directory with calibration JSONs (one per image)
        - mapping: list of (image_filename, calibration_filename) pairs
    """
    def __init__(self, images_dir, calibration_dir, mapping, transform=None):
        self.images_dir = images_dir
        self.calibration_dir = calibration_dir
        self.mapping = mapping  # list of (image_filename, calibration_filename)
        self.transform = transform

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        img_name, calib_name = self.mapping[idx]
        # Use the path directly if it looks like a path, else join with directory
        img_path = img_name if os.path.exists(img_name) or os.path.isabs(img_name) or (os.sep in img_name or '/' in img_name) else os.path.join(self.images_dir, img_name)
        calib_path = calib_name if os.path.exists(calib_name) or os.path.isabs(calib_name) or (os.sep in calib_name or '/' in calib_name) else os.path.join(self.calibration_dir, calib_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        with open(calib_path, 'r') as f:
            calib_json = json.load(f)
        # Try both 'homography' and 'homography_matrix' keys
        H = calib_json.get('homography')
        if H is None:
            H = calib_json.get('homography_matrix')
        # Debug print if format is unexpected
        if isinstance(H, list) and len(H) == 9:
            H = torch.tensor(H, dtype=torch.float32).view(3, 3)
        elif isinstance(H, list) and len(H) == 3 and all(isinstance(row, list) for row in H):
            H = torch.tensor(H, dtype=torch.float32)
        else:
            print(f"[DEBUG] Unexpected homography format in {calib_path}: {H}")
            raise ValueError(f"Unexpected homography format in {calib_path}")
        return image, H
