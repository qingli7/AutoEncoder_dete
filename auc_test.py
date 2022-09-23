import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interpolate
from model import SparseAutoencoder_all
from osr_data import *
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cifar_train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.24705882352941178), # 随机改变图像的亮度对比度和饱和度
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((32, 32), padding=3),
        transforms.ToTensor(),
    ])

## mnist test
# print('mnist test')
# pretrain = 'osr_result_models/osr_mnist_0_ep_41_w_0.10_m_1_s_0_0.9975.pth'
# model = SparseAutoencoder_all(in_channel=1,num_classes=10,feature_dim=512, latent_dim=20).to(device) #
# train_loader, _, ood_val_loader = mnist_data_loader()


# ## svhn test
# print('svhn test')
# pretrain = 'osr_result_models/osr_svhn_0_ep_295_w_0.9_m_0_s_0_0.9660.pth'
# model = SparseAutoencoder_all(in_channel=3,num_classes=10,feature_dim=512, latent_dim=20).to(device) #
# train_loader, _, ood_val_loader = svhn_data_loader()




model = SparseAutoencoder_all(in_channel=3,num_classes=10,feature_dim=512, latent_dim=20).to(device) 

# pretrain = 'osr_result_models/osr_cifar10_0_ep_298_w_0.10_m_1_s_0_0.9388.pth'
# pretrain = 'osr_result_models/osr_cifar10_0_ep_265_w_0.9_m_1_s_0_0.9432.pth'
# pretrain = 'osr_result_models/osr_cifar10_0_ld_20_ep_273_m_0_s_0_0.9397.pth'
## cifar10 test
# print('cifar10 test')
# train_loader, _, ood_val_loader = cifar10_data_loader(row=0)


# pretrain = 'osr_result_models/osr_cifar10_1_ep_169_w_0.10_m_1_s_0_0.9657.pth'
# pretrain = 'osr_result_models/osr_cifar10_1_ld_20_ep_268_m_0_s_0_0.9695.pth'
pretrain = 'osr_result_models/osr_cifar10_1_w_0_ep_265_m_1_s_0_0.9710.pth'
# ## cifar10+10 test
print('cifar10+10 test')
train_loader, _, _ = cifar10_data_loader(row=1)
ood_val_loader, _, _ = cifar100_data_loader(row=1)

## cifar10+50 test
# print('cifar10+50 test')
# train_loader, _, _ = cifar10_data_loader(row=1)
# ood_val_loader, _, _ = cifar100_data_loader(row=1)


## tinyimagenet test
# print('tinyimagenet test')
# pretrain = 'osr_result_models/osr_cifar10_ld_20_ep_220_L1_1_s_0_0.9762.pth'
# model = SparseAutoencoder_all(in_channel=3,num_classes=10,feature_dim=512, latent_dim=20).to(device) #
# test_data = CIFAR10(root='./data/tiny-imagenet-200', train=False, transform=cifar_train_transform, download=True)
# train_loader, val_loader, ood_val_loader = imagenet_data_loader()


print(pretrain)
model.load_state_dict(torch.load(pretrain, map_location='cpu'))
model.eval()
model = model.to(device)
ood_num_scores = len(ood_val_loader)*64 ###  200


def main():
    aurocs, auprs, fprs = [], [], []
    
    # in-distribution Error Rate
    in_score, right_score, wrong_score = get_ood_scores(train_loader, in_dist=True)
    num_right = len(right_score)
    num_wrong = len(wrong_score)
    print('\nError Rate: {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
    
    # OOD Detection
    out_score = get_ood_scores(ood_val_loader)
    print('\nin_score and out_score')
    print(in_score[:10], out_score[:10])
    measures = get_measures(in_score, out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr)
    
    
def print_measures(auroc, aupr, fpr, recall_level=0.95):
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))


def get_ood_scores(loader, in_dist=False):
    _score, _right_score, _wrong_score = [], [], []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_scores and in_dist is False:
                break

            data = data.cuda()
            feat, encoded, decoded = model(data)
            
            # b, k, dim = decoded.shape
            # index_all = torch.tensor([True]).repeat(b, k, dim).cuda()
            # smallest = np.argmin(to_np(torch.sum(torch.square(feat[:, None, :] - decoded), dim=-1)), axis=1)
            # for i in range(b):
            #     for j in range(k):
            #         if j == smallest[i]:
            #             index_all[i][j] = torch.tensor([False]).repeat(dim).cuda()
            # decoded_other = torch.masked_select(decoded, index_all).reshape(b, k-1, dim)
            # mse_other = -torch.sum(torch.square(feat[:, None, :] - decoded_other), dim=-1)
            # loss = to_np(torch.sum(torch.exp(mse_other),dim=-1)) 
            
            # logits = -1e-3* torch.sum(torch.square(feat[:, None, :] - decoded), dim=-1) # L2
            logits = -1e-3* torch.sum(torch.abs(feat[:, None, :] - decoded), dim=-1) # L1
            smax = to_np(F.softmax(logits, dim=1))
            _score.append(np.max(smax, axis=1)) 
            
            # _score.append(np.max(smax, axis=1) + loss) 
            
            
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                _right_score.append(np.max(smax[right_indices], axis=1))
                _wrong_score.append(np.max(smax[wrong_indices], axis=1))
                
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_scores].copy()
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()



def get_measures(_pos, _neg, recall_level=0.95):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    print(pos.shape, neg.shape)
    scores = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(scores), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = metrics.roc_auc_score(labels, scores)
    aupr = metrics.average_precision_score(labels, scores)
    fpr = ErrorRateAt95Recall1(labels, scores, recall_level)
    # fpr, tpr, thresh = metrics.roc_curve(labels, scores)
    # fpr = float(interpolate.interp1d(tpr, fpr)(0.95)) #所有正类中被预测为正类TPR=0.95时，所有反类中被预测为正类FPR的值

    return auroc, aupr, fpr

def ErrorRateAt95Recall1(labels, scores, recall_level):
    # recall_level = 0.95
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]    #降序排列
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_level * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    print('FP:',FP)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    print('TN:',TN)
    
    return float(FP) / float(FP + TN)




if __name__ == '__main__':
    main()