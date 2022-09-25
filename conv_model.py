import torch
import torch.nn as nn
import torchvision
from wide_resnet import WideResNet


class encode_block(nn.Module):
    def __init__(self, in_channel):
        super(encode_block, self).__init__()
        
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channel, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        
    def forward(self, x):
        return self.encoder(x)

class decode_block(nn.Module):
    def __init__(self, in_channel):
        super(decode_block, self).__init__()
        
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
        
    def forward(self, x):
        return self.decoder(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channel, num_classes, feature_dim, latent_dim):
        super(ConvAutoencoder, self).__init__()
        
        self.num_classes = num_classes
        
        self.encoders = nn.ModuleList([encode_block(in_channel) for _ in range(num_classes)])
        self.decoders = nn.ModuleList([decode_block(in_channel) for _ in range(num_classes)])
        

    def forward(self, x):
        """
            feat:(B,C)
            encoded:(B,K,C)
            decoded:(B,K,C)
        """
        encoded = []
        decoded = []
        for i in range(self.num_classes):
            enco = self.encoders[i](x)
            encoded.append(enco)
            deco = self.decoders[i](enco)
            decoded.append(deco)
           
            # breakpoint()

        encoded = torch.stack(encoded, dim=1)
        decoded = torch.stack(decoded, dim=1)

        return encoded, decoded