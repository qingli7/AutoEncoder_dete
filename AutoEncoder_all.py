from xmlrpc.client import Boolean
from pandas import DataFrame
import torch
import argparse
import shutil
from model_all import *
from loss import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data import mnist_data_loader, fashion_mnist_data_loader, cifar10_data_loader, cifar100_data_loader
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='Train Convolutionary Prototype Learning Models')

parser.add_argument('--epochs', default=300, type=int, help='total number of epochs to run')

parser.add_argument('--data_name', default='mnist', type=str, help='dataset name to use')
parser.add_argument('--data_channel', default=1, type=int, help='channel of dataset')
parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')
parser.add_argument('--latent_dim', default=20, type=int, help='latent dimension of prototype feature')
parser.add_argument('--K', default=10, type=int, help='sparse dimension of feature')

# parser.add_argument('--data_name', default='fashion_mnist', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=1, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='cifar10', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')
# parser.add_argument('--latent_dim', default=50, type=int, help='latent dimension of prototype feature')
# parser.add_argument('--K', default=25, type=int, help='sparse dimension of feature')

# parser.add_argument('--data_name', default='cifar100', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=100, type=int, help='class number for the dataset')
# parser.add_argument('--latent_dim', default=100, type=int, help='latent dimension of prototype feature')
# parser.add_argument('--K', default=50, type=int, help='sparse dimension of feature')

parser.add_argument('--feature_dim', default=512, type=int, help='feature dimension of original data')
# parser.add_argument('--latent_dim', default=100, type=int, help='latent dimension of prototype feature')
# parser.add_argument('--K', default=100, type=int, help='sparse dimension of feature')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--cls_weight', type=float, default=0.01, help='ce learning weight')
parser.add_argument('--pl_weight', type=float, default=1e-3, help='pl learning weight')
parser.add_argument('--kl_weight', type=float, default=0.1, help='kl divergence weight') # kl_weight
parser.add_argument('--mse_weight', type=float, default=0.01, help='mse learning weight')
# parser.add_argument('--use_sparse', type=Boolean, default=True, help='sparse autoencoder')
parser.add_argument('--use_sparse', type=Boolean, default=False, help='sparse autoencoder')



args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(): # numclass=0
    print(args)
    if args.data_name=='mnist':
        train_loader, val_loader = mnist_data_loader(batch_size=args.batch_size)
        test_set = MNIST(root='./data/mnist_data', train=False, transform=transforms.ToTensor(), download=True)
    elif args.data_name=='fashion_mnist':
        train_loader, val_loader = fashion_mnist_data_loader(batch_size=args.batch_size)
    elif args.data_name=='cifar10':
        train_loader, val_loader = cifar10_data_loader(batch_size=args.batch_size)
        CIFAR10_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.24705882352941178), 
            transforms.RandomRotation(degrees=5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((32, 32), padding=3),       
            transforms.ToTensor(),
        ])
        test_set = CIFAR10(root='./data/cifar10_data', train=False, transform=CIFAR10_transform, download=True)
    elif args.data_name=='cifar100':
        train_loader, val_loader = cifar100_data_loader(batch_size=args.batch_size)


    model = SparseAutoencoder_all(in_channel=args.data_channel,num_classes=args.num_classes,feature_dim=args.feature_dim, latent_dim=args.latent_dim).to(device) #
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)   #
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 210, 270], gamma=0.1)
    # 150, 210, 270      50, 100, 150
    rho = torch.FloatTensor([0.005 for _ in range(args.K)]).unsqueeze(0).to(device) # args.latent_dim
    # rho = torch.FloatTensor([0.005 for _ in range(args.latent_dim)]).unsqueeze(0).to(device) # args.latent_dim

    cls = CLoss()
    sls = SparseLoss()
    mse = MSELoss()
    temp_acc = 0.8
    # mse = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        train_loss, train_correct, train_total = 0, 0, 0
        val_loss, val_correct, val_total = 0, 0, 0

        model.train()
        # for inputs, targets in tqdm(train_loader):
        for _, (inputs, targets) in enumerate(train_loader):  # tqdm
            inputs = inputs.float().to(device) # [64, 3, 32, 32]
            targets = targets.long().to(device)
            feat, encoded, decoded = model(inputs)
            encoded = torch.tensor([item.cpu().detach().numpy() for item in encoded]).cuda()
            decoded = torch.tensor([item.cpu().detach().numpy() for item in decoded]).cuda()
            # [64, 512] [10, 64, 512] [10, 64, 10] [10, 64, 784]

            MSE_loss = mse(feat,decoded,targets)
            C_loss = args.cls_weight * cls(feat, decoded, targets, args.temp)
            if args.use_sparse:
                Sparse_loss = args.kl_weight * sls(rho, encoded, targets, args.K)
                loss = MSE_loss + C_loss + Sparse_loss
            else: 
                loss = MSE_loss + C_loss
            # print(MSE_loss, C_loss, Sparse_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicts = torch.min(L2_dist(feat, decoded), dim=1)
            # _, predicts = torch.min(torch.sum((feat - decoded), dim=2),dim=0)
            train_loss += loss.item()
            train_total += len(targets)
            train_correct += predicts.to(device).eq(targets).sum().item()

        train_loss = train_loss / len(train_loader)  # / 938 = 60032 / 64
        train_acc =  train_correct / train_total  # / 60,000

        scheduler.step()

        model.eval()
        with torch.no_grad():
            # for inputs, targets in tqdm(val_loader):
            for _, (inputs, targets) in enumerate(val_loader):   # tqdm
                inputs = inputs.float().to(device)# [64, 3, 32, 32]
                targets = targets.long().to(device)
                feat, encoded, decoded = model(inputs)

                MSE_loss = mse(feat,decoded,targets)
                C_loss = args.cls_weight * cls(feat, decoded, targets, args.temp)
                if args.use_sparse:
                    Sparse_loss = args.kl_weight * sls(rho, encoded, targets, args.K)
                    loss = MSE_loss + C_loss + Sparse_loss
                else: 
                    loss = MSE_loss + C_loss
                
                _, predicts = torch.min(L2_dist(feat, decoded), dim=1)
                val_loss = val_loss + loss.item()
                val_total += len(targets)
                val_correct += predicts.to(device).eq(targets).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

        if (epoch % 10 == 0) or (val_acc >= temp_acc): # (val_acc >= temp_acc): val_loss < temp_loss)
            # temp_loss = val_loss
            temp_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/%s/scpn_%s_ld_%d_ep_%d_va_%.4f.pth'
                       % (args.data_name, args.data_name, args.latent_dim, epoch, val_acc))

        print(
            'Epoch : %03d  Train Loss: %.3f | Train Acc: %.3f%% | Val Loss: %.3f | Val Acc: %.3f%%'
            % (epoch, train_loss, 100 * train_acc, val_loss, 100 * val_acc))
            # 'Epoch : %03d  Train Loss: %.3f | Val Loss: %.3f ' % (epoch, train_loss,  val_loss))


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

        x = view_data[i].reshape(1,3,32,32).to(device)
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
    plt.savefig('AE_Sparse_%d.png'%(args.use_sparse), bbox_inches='tight', pad_inches=.25)
    plt.show()


if __name__ == '__main__':

    main()
    # for i in range(10):
        # main(numclass=i)
