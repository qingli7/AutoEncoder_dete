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
        p = torch.softmax(p, dim=1)
        q = torch.softmax(q, dim=1)
        s1 = torch.sum(p * torch.log(p/q))
        s2 = torch.sum((1-p) * torch.log((1-p)/(1-q)))
        return s1 + s2

    def forward(self, rho, encoded, labels, K):
        
        index_dim0 = torch.arange(len(labels))
        encoded = encoded[index_dim0, labels] # (B,dim)
        rho_hat, _ = encoded.topk(K, dim=1,largest=False)
        loss = self.KL_divergence(rho, rho_hat)
        # loss = 0
        # for i in range(len(labels)):
        #     encoded_c = encoded[i,labels[i],:].unsqueeze(dim=0)
        #     rho_hat, _ = torch.topk(encoded_c, K, largest=False)
        #     loss += self.KL_divergence(rho, rho_hat)
        return loss
    
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def L2_dist(self, x, y):
        dist = torch.sqrt(torch.sum(torch.square(x - y)))  # square 按元素求平方
        return dist

    def forward(self, inputs, decoded, labels):
        """
            inputs:(B,dim)
            decoded:(B,K,dim)
            labels:(B)
        """
        index_dim0 = torch.arange(len(labels))
        decoded = decoded[index_dim0, labels]
        MSE_loss = self.L2_dist(inputs, decoded)
        return MSE_loss


class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        
    def L1_dist(self, x, y):
        # x : (B,C)
        # y : (B,K,C)
        # dist : (B,K)
        # x = torch.unsqueeze(x, dim=1)
        # dist = torch.sum(torch.abs(x-y), dim=-1)
        dist = torch.sum(torch.abs(x[:, None, :] - y), dim=-1)
        return dist

    def L2_dist(self, x, y):
        dist = torch.sqrt(torch.sum(torch.square(x[:, None, :] - y), dim=-1))  # square 按元素求平方
        return dist

    def forward(self, feat, feat2, labels, temp):
        # logits = -temp * self.L1_dist(feat, feat2) 
        logits = -temp * self.L2_dist(feat, feat2)
        loss = self.loss(logits, labels)
        return loss


if __name__ == '__main__':
    features = torch.rand((30, 2))
    prototypes = torch.rand((10, 2))
    labels = torch.rand((30,)).long()
    criterion = CLoss()
    loss = criterion(features, prototypes, labels)
    print(loss)
