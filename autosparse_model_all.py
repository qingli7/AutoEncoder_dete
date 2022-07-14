import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def spatial_sparsity(encoded):
        position = torch.zeros(encoded.shape)
        # print(encoded)
        b = encoded.shape[0]
        c = encoded.shape[3]

        encoded_t = torch.transpose(encoded, [0, 3, 1, 2])  # b, c, h, w
        encoded_r = encoded_t.reshape([b, c, -1])  # b, c, h*w

        th, _ = torch.topk(encoded_r, 1)  # b, c, 1
        th_r = th.reshape([b, 1, 1, c])  # b, 1, 1, c

        drop = torch.where(encoded < th_r, torch.full_like(position,1), position)

        # spatially dropped and top element (winners)
        return encoded * drop, th.reshape([b, c])  # b, c


class SparseAutoencoder(nn.Module):
    def __init__(self, in_channel, num_classes, feature_dim, latent_dim):
        super(SparseAutoencoder, self).__init__()

        cls = torchvision.models.resnet18(pretrained=True)
        cls.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=3)
        self.cls = nn.Sequential(*list(cls.children())[:-1])
        
        if in_channel == 1:
            recover_dim =  1*28*28
        if in_channel==3:
            recover_dim =  3*32*32

        self.fc1 = nn.Linear(feature_dim, latent_dim) # *num_classes
        self.fc2 = nn.Linear(latent_dim, recover_dim)  # 1*28*28;  (64*16*16, feature_dim)  # 28-3+6+1=32, 32/2=16;

        self.fc = nn.Linear(latent_dim, num_classes)
        
        
        # self.prototypes = nn.Parameter(
        #     num_classes * torch.randn((num_classes, latent_dim)) - num_classes / 2., requires_grad=True)

        
    def forward(self, x):
        encoded = self.cls(x).reshape(x.shape[0], -1)
        encoded = self.fc1(encoded)
        encoded = torch.sigmoid(encoded)
        
        decoded = self.fc2(encoded)
        return encoded, decoded
