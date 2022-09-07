import tqdm
import torch
import numpy as np
from model import SparseAutoencoder_all
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100

from data import mnist_data_loader, fashion_mnist_data_loader, cifar10_data_loader, cifar100_data_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# plot ori/sparse features
def show_compare_features(data_name, in_channel, num_classes):
    pretrain = 'results_models/gcpn_fashion_mnist_ep_135_ld_2_va_0.9351.pth'
    model = SparseAutoencoder_all(in_channel=in_channel,num_classes=num_classes,feature_dim=512, latent_dim=20).to(device) #
    model.load_state_dict(torch.load(pretrain, map_location='cpu'))
    model.eval()
    model = model.to(device)
    
    # plot ori/sparse features of selected dataset
    cifar_train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.24705882352941178), # 随机改变图像的亮度对比度和饱和度
            transforms.RandomRotation(degrees=5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((32, 32), padding=3),
            transforms.ToTensor(),
        ])
    if data_name=='mnist':
            test_set = MNIST(root='./data/mnist_data', train=False, transform=transforms.ToTensor(), download=True)
            image_shape = (1,1,28,28)
    elif data_name=='cifar10':
            train_loader, val_loader = cifar10_data_loader(batch_size=64)
            test_set = CIFAR10(root='./data/cifar10_data', train=False, transform=cifar_train_transform, download=True)
            image_shape = (1,3,32,32)
    elif data_name=='cifar100':
            train_loader, val_loader = cifar100_data_loader(batch_size=64)

    N_ROWS = 4
    N_COLS = 8
    view_data = [test_set[i][0] for i in range(N_ROWS * N_COLS)]
    view_target = [test_set[i][1] for i in range(N_ROWS * N_COLS)]
    plt.figure(figsize=(20, 4))
    plt.figure(figsize=(8,8))
    for i in range(N_ROWS * N_COLS):
        # original image
        r = i // N_COLS
        c = i % N_COLS + 1  # % 是求余数，// 是求商+向下取整
        ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c)

        x = view_data[i].reshape(image_shape).to(device)
        feat, encoded, decoded = model(x)
        plt.imshow(feat.detach().cpu().squeeze().numpy().reshape(16,32))

        # plt.imshow(view_data[i].squeeze())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstructed image
        ax = plt.subplot(2 * N_ROWS, N_COLS, 2 * r * N_COLS + c + N_COLS)
        plt.imshow(decoded[:,view_target[i],:].detach().cpu().squeeze().numpy().reshape(16,32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('AE_Sparse_%s.png'%(data_name), bbox_inches='tight', pad_inches=.25)
    plt.show()


def get_feat():
    pretrain = 'results_models/gcpn_fashion_mnist_ep_135_ld_2_va_0.9351.pth'
    model = SparseAutoencoder_all(num_classes=10, in_channel=1, latent_dim=2).to(device)
    model.load_state_dict(torch.load(pretrain, map_location='cpu'))
    model.eval()
    model = model.to(device)

    # c_train, c_test = mnist_data_loader()
    c_train, c_test = fashion_mnist_data_loader()
    # c_train, c_test = cifar10_data_loader()
    # c_train, c_test = cifar100_data_loader()
    
    feats, labels = list(), list()
    # for img, label in tqdm.tqdm(list(c_train)):  # list( tup )
    for img, label in tqdm.tqdm(list(c_test)):  # list( tup )
        img = img.float().to(device)
        feat = model(img)
        feats.append(feat.detach().cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0) # axis=0表示沿着数组垂直方向进行拼接


def show_classification(embedding, c):
    plt.figure(figsize=(6, 6))
    # plt.title('Scatter Plot')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    for i in range(10):
        index = np.where(c == i)[0].tolist()
        plt.scatter(embedding[index, 0], embedding[index, 1], s=5)
    plt.legend([str(i) for i in range(10)], loc='upper right')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('gcpn_fashion_mnist_ep_135_ld_2_va_0.9351.png', bbox_inches='tight', pad_inches=.05)
    plt.close()
    # plt.show()
 


if __name__ == '__main__':
    feats, labels = get_feat()
    show_classification(feats, labels)
    print('Saved figure!')
    
    show_compare_features('mnist', 1, 10)
    # show_compare_features('cifar10', 3, 10)
    # show_compare_features('cifar100', 3, 100)
    # show_compare_features('svhn', 3, 10)