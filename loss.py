import torch
import torch.nn as nn
import torch.nn.functional as F

def L2_dist(x, y):
    # x : shape [batch, dim]
    # y : shape [num_classes, dim]
    # dist : [batch, num_classes]
    dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
    return dist



class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()

    def KL_divergence(self, p, q):
        p = F.softmax(p, dim=1) # 为什么是log?
        q = F.softmax(q, dim=1)
        s1 = torch.sum(p * torch.log(p/q))
        s2 = torch.sum((1-p) * torch.log((1-p)/(1-q)))
        return s1 + s2

    def forward(self, rho, encoded, labels, K):
        loss = 0
        index_dim0 = torch.arange(len(labels))
        encoded_c = encoded[index_dim0, labels, :].unsqueeze(dim=0)
        # for i in range(len(labels)):
        # encoded_c = encoded[i,labels[i],:].unsqueeze(dim=0)
        rho_hat, _ = torch.topk(encoded_c, K, largest=False)
        loss = loss + self.KL_divergence(rho, rho_hat)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def L2_dist(self, x, y):
        # x : (B,C)
        # y : (B,C)
        # dist : scalar
        dist = torch.sqrt(torch.sum(torch.square(x - y)))  # square 按元素求平方
        return dist

    def forward(self, inputs, decoded, labels):
        """
            inputs:(B,C,H,W)
            decoded:(B,K,C,H,W)
            labels:(B)
        """
        # breakpoint()
        # decoded = decoded.reshape(len(inputs),10,-1)   #  [64, 10, 784]
        index_dim0 = torch.arange(len(labels))
        decoded = decoded[index_dim0, labels]
        dist = self.L2_dist(inputs, decoded)
        # gather利用index来索引input特定位置的数值，unsqueeze(-1)拆分元素
        # select_dist = torch.gather(dist, dim=1, index=labels.unsqueeze(-1).to(torch.int64)) #(B,1)
        # MSE_loss = torch.mean(dist)
        # print(MSE_loss)
        return dist


class CLoss(nn.Module):
    def __init__(self, t=1):
        super(CLoss, self).__init__()
        self.t = t
        self.loss = nn.CrossEntropyLoss()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))  # square 按元素求平方
        return dist

    def forward(self, feat, feat2, labels):
        # feat2 = feat2.reshape(len(feat),10,-1)
        logits = -self.L2_dist(feat, feat2) / self.t
        # print(logits,feat2.shape, feat.shape)
        loss = self.loss(logits, labels)
        return loss


class PLoss(nn.Module):
    def __init__(self):
        super(PLoss, self).__init__()

    def L2_dist(self, x, y):
        # x : shape [batch, dim], 64 x 512
        # y : shape [num_classes, dim], C x 512
        # dist : [batch, num_classes], 64 x C
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))
        return dist

    def forward(self, feat, prototype, label):
        dist = self.L2_dist(feat, prototype)
        # gather利用index来索引input特定位置的数值，unsqueeze(-1)拆分元素
        pos_dist = torch.gather(dist, dim=1, index=label.unsqueeze(-1).to(torch.int64))
        pl = torch.mean(pos_dist)
        return pl


if __name__ == '__main__':
    features = torch.rand((30, 2))
    prototypes = torch.rand((10, 2))
    labels = torch.rand((30,)).long()
    criterion = CLoss()
    loss = criterion(features, prototypes, labels)
    print(loss)
