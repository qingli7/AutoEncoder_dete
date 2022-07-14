import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from autosparse_model_all import SparseAutoencoder

from data import mnist_data_loader, fashion_mnist_data_loader, cifar10_data_loader, cifar100_data_loader
from AutoEncoder_all import args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_feat():
    pretrain = 'result_new/scpn_fashion_mnist_ep_68_K_10_va_0.8944.pth'
    model = SparseAutoencoder(in_channel=1, num_classes=10, feature_dim=512, latent_dim=512).to(device)
    model.load_state_dict(torch.load(pretrain, map_location='cpu'))
    model.eval()

    # train_loader, val_loader = mnist_data_loader(batch_size=args.batch_size)
    train_loader, val_loader = fashion_mnist_data_loader(batch_size=args.batch_size)
    # train_loader, val_loader = cifar10_data_loader(batch_size=args.batch_size)
    # train_loader, val_loader = cifar100_data_loader(batch_size=args.batch_size)

    feats, labels = list(), list()
    for img, label in tqdm.tqdm(list(val_loader)):  # list( tup )
        img = img.float().to(device)
        feat, _ = model(img)
        feat = model.fc1(feat)
        feats.append(feat.detach().cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0) # axis=0表示沿着数组垂直方向进行拼接


def show(embedding, c):

    plt.figure(figsize=(6, 6))
    # plt.title('Scatter Plot')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    for i in range(10):  ###
        index = np.where(c == i)[0].tolist()
        plt.scatter(embedding[index, 0], embedding[index, 1], s=5)
    plt.legend([str(i) for i in range(10)], loc='upper left')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('Sparse_fashion_mnist_ep_68_K_10_va_0.8944.png', bbox_inches='tight', pad_inches=.05) 
    plt.close()
    # plt.show()
 


if __name__ == '__main__':
    feats, labels = get_feat()
    print('feats, labels:',feats.shape, labels.shape)
    show(feats, labels)
    print('Saved figure!')