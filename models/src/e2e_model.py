import torch
import torch.nn as nn
import torchvision.models as models

class BallToCourtNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        # Use a ResNet backbone (remove final fc layer)
        if backbone == 'resnet18':
            self.cnn = models.resnet18(pretrained=pretrained)
            num_feats = self.cnn.fc.in_features
        elif backbone == 'resnet34':
            self.cnn = models.resnet34(pretrained=pretrained)
            num_feats = self.cnn.fc.in_features
        else:
            raise ValueError('Unsupported backbone')
        self.cnn.fc = nn.Identity()  # Remove final layer
        # Final regressor: [CNN features + pixel_x + pixel_y] -> [court_x, court_y]
        self.regressor = nn.Sequential(
            nn.Linear(num_feats + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, image, pixel_coords):
        feats = self.cnn(image)  # [B, num_feats]
        x = torch.cat([feats, pixel_coords], dim=1)  # [B, num_feats+2]
        out = self.regressor(x)  # [B, 2]
        return out

# Example usage:
# model = BallToCourtNet(backbone='resnet18', pretrained=True)
# output = model(images, pixel_coords)
