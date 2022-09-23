from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, ImageNet
from torchvision import datasets
from data_aug import *
from osr_split import splits
import torch
import os


def cifar_filter(self, known):
    datas, targets = np.array(self.data), np.array(self.targets)
    mask, new_targets = [], []
    for i in range(len(targets)):
        if targets[i] in known:
            mask.append(i)
            new_targets.append(known.index(targets[i]))
    self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)
    
    
def mnist_filter(self, known):
    targets = self.targets.data.numpy()
    mask, new_targets = [], []
    for i in range(len(targets)):
        if targets[i] in known:
            mask.append(i)
            new_targets.append(known.index(targets[i]))
    self.targets = np.array(new_targets)
    mask = torch.tensor(mask).long()
    self.data = torch.index_select(self.data, 0, mask)
      

def svhn_filter(self, known):     
    targets = np.array(self.labels)
    mask, new_targets = [], []
    for i in range(len(targets)):
        if targets[i] in known:
            mask.append(i)
            new_targets.append(known.index(targets[i]))
    self.data, self.labels = self.data[mask], np.array(new_targets)


def mnist_data_loader(mnist_folder='./data/mnist_data', batch_size=64, row=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = MNIST(mnist_folder, train=True, download=True, transform=transform)
    test_set = MNIST(mnist_folder, train=False, download=True, transform=transform)
    out_set = MNIST(mnist_folder, train=False, download=True, transform=transform)
    known = splits['mnist'][row]
    unknown = list(set(list(range(0, 10))) - set(known))
    mnist_filter(train_set, known)
    mnist_filter(test_set, known)
    mnist_filter(out_set, unknown)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True) #, num_workers=4)
    out_loader = DataLoader(out_set, batch_size=batch_size, shuffle=True) #, num_workers=4)
    
    return train_loader, test_loader, out_loader



def cifar10_data_loader(folder='./data/cifar10_data', batch_size=64, transform=True, row=0):
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.24705882352941178), # 随机改变图像的亮度对比度和饱和度
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((32, 32), padding=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]), 
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]), 
    ])
    train_set = CIFAR10(folder, train=True, download=True, transform=train_transform)
    test_set = CIFAR10(folder, train=False, download=True, transform=transform)
    out_set = CIFAR10(folder, train=False, download=True, transform=transform)
    known = splits['cifar10'][row]
    unknown = list(set(list(range(0, 10))) - set(known))
    cifar_filter(train_set, known)
    cifar_filter(test_set, known)
    cifar_filter(out_set, unknown)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    out_loader = DataLoader(out_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, out_loader


def cifar100_data_loader(folder='./data/cifar100_data', batch_size=64, row=0):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomCrop((32, 32), padding=3),
        transforms.ToTensor(),
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = CIFAR100(folder, train=True, download=True, transform=train_transform)
    test_set = CIFAR100(folder, train=False, download=True, transform=transform)
    out_set = CIFAR100(folder, train=False, download=True, transform=transform)
    known = splits['cifar100'][row]
    unknown = list(set(list(range(0, 100))) - set(known))
    cifar_filter(train_set, known)
    cifar_filter(test_set, known)
    cifar_filter(out_set, unknown)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    out_loader = DataLoader(out_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, out_loader


def svhn_data_loader(folder='./data/svhn_data', batch_size=64, row=0):
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.24705882352941178), # 随机改变图像的亮度对比度和饱和度
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((32, 32), padding=3),
        transforms.ToTensor(),
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = SVHN(folder, split='train', download=True, transform=train_transform)
    test_set = SVHN(folder, split='test', download=True, transform=transform)
    out_set = SVHN(folder, split='test', download=True, transform=transform)
    known = splits['svhn'][row]
    unknown = list(set(list(range(0, 100))) - set(known))
    svhn_filter(train_set, known)
    svhn_filter(test_set, known)
    svhn_filter(out_set, unknown)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    out_loader = DataLoader(out_set, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, out_loader


def imagenet_data_loader(folder='./data/tiny-imagenet-200', batch_size=64):
    # num_label = 200
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), 
                             (0.2770, 0.2691, 0.2821))
    ])
    test_transform = transforms.Compose([
        # transforms.Resize(32), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4802, 0.4481, 0.3975), 
                             (0.2770, 0.2691, 0.2821))
    ])
    train_set = datasets.ImageFolder(root=os.path.join(folder, 'train'), transform=train_transform)
    test_set = datasets.ImageFolder(root=os.path.join(folder, 'val'), transform=test_transform)
    out_set = datasets.ImageFolder(root=os.path.join(folder, 'val'), transform=test_transform)
    print(train_set)
    known = splits['tiny_imagenet'][0]
    unknown = list(set(list(range(0, 200))) - set(known))
    svhn_filter(train_set, known)
    svhn_filter(test_set, known)
    svhn_filter(out_set, unknown)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    out_loader = DataLoader(out_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, out_loader

# # # white/real/blind/hybrid noisy image denoising
# train_transform_noise=transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=5),
#     transforms.RandomCrop((32, 32), padding=3),
        
#     PepperSaltNoise(p=0.1),
#     ColorPointNoise(p=0.1),
#     GaussianNoise(p=0.1),
#     Mosaic(p=0.1),
#     RGBShuffle(p=0.05),
#     Rotate(p=0.1),
#     HFlip(p=0.1),
#     VFlip(p=0.05),
#     RandomCut(p=0.1),
#     MotionBlur(p=0.1),
#     GaussianBlur(p=0.01),
#     Blur(p=0.01),
#     Rain(p=0.1),
    
#     # transforms.Resize(256),
#     # transforms.Resize(224),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
# ])



if __name__ == '__main__':
    data_train, data_test = mnist_data_loader()  # checked
    data_train, data_test = cifar10_data_loader()  # checked
    data_train, data_test = cifar100_data_loader()  # checked
    for x, y in data_train:
        print(x.shape, y.shape, x.max(), x.min())