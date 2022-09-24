import torch
import torch.nn as nn
import torchvision
from wide_resnet import WideResNet


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channel, num_classes, feature_dim, latent_dim):
        super(ConvAutoencoder, self).__init__()
        
        # self.cls = WideResNet(depth=28, in_channel=in_channel,num_classes=num_classes, feature_dim=feature_dim, widen_factor=10)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh()
        )

        self.encode = nn.ModuleList([self.encoder for _ in range(num_classes)])
        self.decode = nn.ModuleList([self.decoder for _ in range(num_classes)])
        

    def forward(self, x):
        """
            feat:(B,C)
            encoded:(B,K,C)
            decoded:(B,K,C)
        """
        encoded = []
        decoded = []
        for i in range(len(self.encode)):
            enco = x
            for layer in self.encode[i]:
                # print(i, layer)
                enco = layer(enco)
            encoded.append(enco)
            
            deco = enco
            for layer in self.decode[i]:
                deco = layer(deco)
            decoded.append(deco)
            breakpoint()
            # print(i,':',x[0][0][1][1],enco[0][0][1][1],deco[0][0][1][1])

        encoded = torch.stack(encoded, dim=1)
        decoded = torch.stack(decoded, dim=1)

        return encoded, decoded