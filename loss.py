import torch
import torch.nn as nn

def L2_dist(x, y):
    # x : shape [batch, dim]
    # y : shape [num_classes, dim]
    # dist : [batch, num_classes]
    dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1)) 
    return dist


def KL_divergence(p, q):
    p = torch.softmax(p, dim=1)
    q = torch.softmax(q, dim=1)
    s1 = torch.sum(p * torch.log(p/q))
    s2 = torch.sum((1-p) * torch.log((1-p)/(1-q)))
    return s1 + s2


def MSELoss(inputs, decoded):
    batch, channel, w, h = inputs.shape
    MSE_loss = (inputs.reshape(batch,-1) - decoded) ** 2 
    MSE_loss =  torch.sqrt(MSE_loss.view(1, -1).sum(1)) / (batch*channel*w*h)
    return MSE_loss
    

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

    def forward(self, feats, prototypes, labels):
        logits = -self.L2_dist(feats, prototypes) / self.t
        # print('logits: ', logits.dtype, logits.shape)
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
