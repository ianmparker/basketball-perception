
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch import nn, optim
from e2e_dataset import BallCourtE2EDataset
from e2e_model import BallToCourtNet

def main():
    # Config
    CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/e2e_training_data_clean.csv'))
    IMG_DIR = '../../data/e2e_frames/'
    BATCH_SIZE = 32
    EPOCHS = 20
    VAL_SPLIT = 0.15
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BallCourtE2EDataset(CSV_PATH, IMG_DIR, transform=transform)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = BallToCourtNet(backbone='resnet18', pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, pixel_coords, court_coords in train_loader:
            images = images.to(DEVICE)
            pixel_coords = pixel_coords.to(DEVICE)
            court_coords = court_coords.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images, pixel_coords)
            loss = criterion(preds, court_coords)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        val_error = 0
        with torch.no_grad():
            for images, pixel_coords, court_coords in val_loader:
                images = images.to(DEVICE)
                pixel_coords = pixel_coords.to(DEVICE)
                court_coords = court_coords.to(DEVICE)
                preds = model(images, pixel_coords)
                loss = criterion(preds, court_coords)
                val_loss += loss.item() * images.size(0)
                # Compute mean Euclidean error in court space
                error = torch.norm(preds - court_coords, dim=1).sum().item()
                val_error += error
        val_loss /= len(val_loader.dataset)
        val_error /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean Error (ft): {val_error:.3f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/e2e_model.pth')
    print('Model saved to checkpoints/e2e_model.pth')

if __name__ == '__main__':
    main()
