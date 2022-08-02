'''
Author: hushuwang hushuwang2019@ia.ac.cn
Date: 2022-08-01 11:37:04
LastEditors: hushuwang hushuwang2019@ia.ac.cn
LastEditTime: 2022-08-02 11:37:12
FilePath: /AutoEncoder_dete/model_all.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SparseAutoencoder_all(nn.Module):
    def __init__(self, in_channel, num_classes, feature_dim, latent_dim):
        super(SparseAutoencoder_all, self).__init__()
        self.in_channel = in_channel
        self.classes = num_classes
        cls = torchvision.models.resnet18(pretrained=True)
        cls.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=3)
        self.cls = nn.Sequential(*list(cls.children())[:-1])
        cnt = 0
        for child in self.cls.children():
            cnt += 1
            if cnt>1:
                for param in child.parameters():
                    param.requires_grad = False
        # if in_channel == 1:
        #     recover_dim =  1*28*28
        # if in_channel==3:
        #     recover_dim =  3*32*32
        # recover_dim = feature_dim

        self.fc1 = nn.ModuleList([nn.Linear(feature_dim, latent_dim) for _ in range(num_classes)])
        self.fc2 = nn.ModuleList([nn.Linear(latent_dim, feature_dim) for _ in range(num_classes)])


    def forward(self, x):
        """
            feat:(B,C)
            encoded:(B,K,C)
            decoded:(B,K,C)
        """
        feat = self.cls(x).reshape(x.shape[0], -1)

        encoded = []
        decoded = []
        for i in range(len(self.fc1)):
            encoded.append(torch.sigmoid(self.fc1[i](feat)))
            decoded.append(torch.sigmoid(self.fc2[i](encoded[i])))
        encoded = torch.stack(encoded, dim=1)
        decoded = torch.stack(decoded, dim=1)
        # breakpoint()

        return feat, encoded, decoded


class SparseAutoencoder(nn.Module):
    def __init__(self, in_channel, feature_dim, latent_dim):
        super(SparseAutoencoder, self).__init__()
        # if in_channel == 1:
        #     recover_dim =  1*28*28
        # if in_channel==3:
        #     recover_dim =  3*32*32
        recover_dim = feature_dim
        self.encoder = nn.Linear(recover_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, recover_dim)

        cls = torchvision.models.resnet18(pretrained=True)
        cls.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=3)
        self.cls = nn.Sequential(*list(cls.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        feat = self.fc(self.cls(x).reshape(x.shape[0], -1))
        encoded = torch.sigmoid(self.encoder(feat))
        decoded = torch.sigmoid(self.decoder(encoded))
        return feat, encoded, decoded