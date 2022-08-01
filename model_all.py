import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SparseAutoencoder_all(nn.Module):
    def __init__(self, in_channel, num_classes, feature_dim, latent_dim):
        super(SparseAutoencoder_all, self).__init__()

        cls = torchvision.models.resnet18(pretrained=True)
        cls.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=3)
        self.cls = nn.Sequential(*list(cls.children())[:-1])

        # if in_channel == 1:
        #     recover_dim =  1*28*28
        # if in_channel==3:
        #     recover_dim =  3*32*32
        recover_dim = feature_dim

        self.fc1_0 = nn.Linear(recover_dim, latent_dim)
        self.fc2_0 = nn.Linear(latent_dim, recover_dim)
        self.fc1_1 = nn.Linear(recover_dim, latent_dim)
        self.fc2_1 = nn.Linear(latent_dim, recover_dim)
        self.fc1_2 = nn.Linear(recover_dim, latent_dim)
        self.fc2_2 = nn.Linear(latent_dim, recover_dim)
        self.fc1_3 = nn.Linear(recover_dim, latent_dim)
        self.fc2_3 = nn.Linear(latent_dim, recover_dim)
        self.fc1_4 = nn.Linear(recover_dim, latent_dim)
        self.fc2_4 = nn.Linear(latent_dim, recover_dim)
        self.fc1_5 = nn.Linear(recover_dim, latent_dim)
        self.fc2_5 = nn.Linear(latent_dim, recover_dim)
        self.fc1_6 = nn.Linear(recover_dim, latent_dim)
        self.fc2_6 = nn.Linear(latent_dim, recover_dim)
        self.fc1_7 = nn.Linear(recover_dim, latent_dim)
        self.fc2_7 = nn.Linear(latent_dim, recover_dim)
        self.fc1_8 = nn.Linear(recover_dim, latent_dim)
        self.fc2_8 = nn.Linear(latent_dim, recover_dim)
        self.fc1_9 = nn.Linear(recover_dim, latent_dim)
        self.fc2_9 = nn.Linear(latent_dim, recover_dim)
        self.fc1 = nn.ModuleList([self.fc1_0, self.fc1_1, self.fc1_2, self.fc1_3, self.fc1_4, self.fc1_5, self.fc1_6, self.fc1_7, self.fc1_8, self.fc1_9])
        self.fc2 = nn.ModuleList([self.fc2_0, self.fc2_1, self.fc2_2, self.fc2_3, self.fc2_4, self.fc2_5, self.fc2_6, self.fc2_7, self.fc2_8, self.fc2_9])


    def forward(self, x):
        feat = self.cls(x).reshape(x.shape[0], -1)

        encoded = []
        decoded = []
        for i in range(len(self.fc1)):
            encoded.append(torch.sigmoid(self.fc1[i](feat)))
            decoded.append(torch.sigmoid(self.fc2[i](encoded[i])))
        encoded = torch.stack(encoded, dim=1)
        decoded = torch.stack(decoded, dim=1)
        # breakpoint()
        
        # encoded_0 = torch.sigmoid(self.fc1[0](feat))
        # decoded_0 = torch.sigmoid(self.fc2[0](encoded_0))
        # encoded_1 = torch.sigmoid(self.fc1[0](feat))
        # decoded_1 = torch.sigmoid(self.fc2[0](encoded_1))
        # encoded_2 = torch.sigmoid(self.fc1[0](feat))
        # decoded_2 = torch.sigmoid(self.fc2[0](encoded_2))
        # encoded_3 = torch.sigmoid(self.fc1[0](feat))
        # decoded_3 = torch.sigmoid(self.fc2[0](encoded_3))
        # encoded_4 = torch.sigmoid(self.fc1[0](feat))
        # decoded_4 = torch.sigmoid(self.fc2[0](encoded_4))
        # encoded_5 = torch.sigmoid(self.fc1[0](feat))
        # decoded_5 = torch.sigmoid(self.fc2[0](encoded_5))
        # encoded_6 = torch.sigmoid(self.fc1[0](feat))
        # decoded_6 = torch.sigmoid(self.fc2[0](encoded_6))
        # encoded_7 = torch.sigmoid(self.fc1[0](feat))
        # decoded_7 = torch.sigmoid(self.fc2[0](encoded_7))
        # encoded_8 = torch.sigmoid(self.fc1[0](feat))
        # decoded_8 = torch.sigmoid(self.fc2[0](encoded_8))
        # encoded_9 = torch.sigmoid(self.fc1[0](feat))
        # decoded_9 = torch.sigmoid(self.fc2[0](encoded_9))
        # encoded = [encoded_0, encoded_1, encoded_2, encoded_3, encoded_4, encoded_5, encoded_6, encoded_7, encoded_8, encoded_9]
        # decoded = [decoded_0, decoded_1, decoded_2, decoded_3, decoded_4, decoded_5, decoded_6, decoded_7, decoded_8, decoded_9]
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