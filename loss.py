import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def L1_dist(x):
    dist = torch.sqrt(torch.sum(torch.abs(x), dim=-1))
    return dist

def L2_dist(x, y):
    # x : shape [batch, dim]
    # y : shape [num_classes, dim]
    # dist : [batch, num_classes]
    dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
    return dist


class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        return dist

    def forward(self, inputs, recovers, label):
        # print(feat.shape, prototype.shape)
        inputs = inputs.reshape(inputs.shape[0], -1)
        dist = self.L2_dist(inputs, recovers)
        # gather利用index来索引input特定位置的数值，unsqueeze(-1)拆分元素
        pos_dist = torch.gather(dist, dim=1, index=label.unsqueeze(-1).to(torch.int64))
        sl = torch.mean(pos_dist)
        return sl
   
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def L2_dist(self, x, y):
        dist = torch.sqrt(torch.sum(torch.square(x - y)))  # square 按元素求平方
        return dist

    def forward(self, inputs, decoded, labels, obj=True):
        """
            inputs:(B,dim)
            decoded:(B,K,dim)
            labels:(B)
        """
        if obj:
            index_dim0 = torch.arange(len(labels))
            decoded_tar = decoded[index_dim0, labels]
            loss = self.L2_dist(inputs, decoded_tar)/len(labels)
        else:
            b, k, dim = decoded.shape
            index_all = torch.tensor([True]).repeat(b, k, dim).cuda()
            for i in range(b):
                index_all[i][labels[i]] = torch.tensor([False]).repeat(dim).cuda()
            decoded_other = torch.masked_select(decoded, index_all).reshape(b, k-1, dim)
            # mse_other = -torch.sum(torch.abs(inputs[:, None, :] - decoded_other), dim=-1)
            mse_other = -torch.sum(torch.square(inputs[:, None, :] - decoded_other), dim=-1)
            loss = torch.sum(torch.exp(mse_other)) 
                
        return loss


class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        
    def L1_dist(self, x, y):
        # x : (B,C)
        # y : (B,K,C)
        # dist : (B,K)
        # x = torch.unsqueeze(x, dim=1) # [64, 512] --> [64, 1, 512]
        # dist = torch.sum(torch.abs(x-y), dim=-1)
        dist = torch.sum(torch.abs(x[:, None, :] - y), dim=-1)
        return dist

    def forward(self, feat, feat2, labels, temp):
        logits = -temp * self.L1_dist(feat, feat2)
        loss = self.loss(logits, labels)
        return loss



if __name__ == '__main__':
    features = torch.rand((30, 2))
    prototypes = torch.rand((10, 2))
    labels = torch.rand((30,)).long()
    criterion = CLoss()
    loss = criterion(features, prototypes, labels)
    print(loss)
