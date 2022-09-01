import torch
import torch.nn as nn
import torchvision
from wideresnet import WideResNet


class SparseAutoencoder_all(nn.Module):
    def __init__(self, in_channel, num_classes, feature_dim, latent_dim):
        super(SparseAutoencoder_all, self).__init__()

        # cls = torchvision.models.resnet18(pretrained=True)
        # cls.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=3)
        # self.cls = nn.Sequential(*list(cls.children())[:-1])
        
        self.cls = WideResNet(depth=28, num_classes=num_classes, feature_dim=feature_dim, widen_factor=10)

        self.fc1 = nn.ModuleList([nn.Linear(feature_dim, latent_dim) for _ in range(num_classes)])
        self.fc2 = nn.ModuleList([nn.Linear(latent_dim, feature_dim) for _ in range(num_classes)])

    def forward(self, x):
        """
            feat:(B,C)
            encoded:(B,K,C)
            decoded:(B,K,C)
        """
        # feat = self.cls(x).reshape(x.shape[0], -1)
        feat = self.cls(x)
        
        # breakpoint()
        encoded = []
        decoded = []
        for i in range(len(self.fc1)):
            encoded.append(torch.sigmoid(self.fc1[i](feat)))
            decoded.append(torch.sigmoid(self.fc2[i](encoded[i])))
            # encoded.append(torch.tanh(self.fc1[i](feat)))
            # decoded.append(torch.tanh(self.fc2[i](encoded[i])))
        encoded = torch.stack(encoded, dim=1)
        decoded = torch.stack(decoded, dim=1)

        return feat, encoded, decoded