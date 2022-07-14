from xmlrpc.client import Boolean
from pandas import DataFrame
import torch
import argparse
import shutil
from autosparse_model_all import SparseAutoencoder
from loss import L2_dist,KL_divergence, CLoss, PLoss
from tqdm import tqdm
from tensorboardX import SummaryWriter
from data import mnist_data_loader, fashion_mnist_data_loader, cifar10_data_loader, cifar100_data_loader
from torch import nn
import matplotlib.pyplot as plt 
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Train Convolutionary Prototype Learning Models')

parser.add_argument('--epochs', default=300, type=int, help='total number of epochs to run')

parser.add_argument('--data_name', default='mnist', type=str, help='dataset name to use')
parser.add_argument('--data_channel', default=1, type=int, help='channel of dataset')
parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='fashion_mnist', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=1, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='cifar10', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='cifar100', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=100, type=int, help='class number for the dataset')

parser.add_argument('--feature_dim', default=512, type=int, help='feature dimension of original data')
parser.add_argument('--latent_dim', default=100, type=int, help='latent dimension of prototype feature')
parser.add_argument('--K', default=10, type=int, help='basis dimension of feature') 
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--cls_weight', type=float, default=0.1, help='ce learning weight')
parser.add_argument('--pl_weight', type=float, default=1e-3, help='pl learning weight')
parser.add_argument('--kl_weight', type=float, default=0.01, help='kl divergence weight') # kl_weight
parser.add_argument('--mse_weight', type=float, default=0.01, help='mse learning weight')
parser.add_argument('--use_sparse', type=Boolean, default=False, help='sparse autoencoder')



args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(): # numclass=0
    print(args)
    # print('numclass:',numclass)
    train_loader, val_loader = mnist_data_loader(batch_size=args.batch_size)
    # train_loader, val_loader = fashion_mnist_data_loader(batch_size=args.batch_size)
    # train_loader, val_loader = cifar10_data_loader(batch_size=args.batch_size)
    # train_loader, val_loader = cifar100_data_loader(batch_size=args.batch_size)
    
    
    # model = resnet18_gcpl(num_classes=args.num_classes, in_channel=args.data_channel, latent_dim=args.latent_dim).to(device)
    model = SparseAutoencoder(in_channel=args.data_channel, num_classes=args.num_classes, feature_dim=args.feature_dim, latent_dim=args.latent_dim).to(device) #
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)   # 
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 210, 270], gamma=0.1)
    rho = torch.FloatTensor([0.005 for _ in range(100)]).unsqueeze(0).to(device) # args.latent_dim
    
    cls = CLoss()
    pl = PLoss()
    temp_acc = 0.8
    temp_loss = 0.5
    
    loss_plot = []
    loss_plot1 = []
    acc_plot = []
    acc_plot1 = []
    for epoch in range(args.epochs):
        train_loss, train_correct, train_total = 0, 0, 0
        val_loss, val_correct, val_total = 0, 0, 0

        model.train()
        # for inputs, targets in tqdm(train_loader):
        for _, (inputs, targets) in enumerate(train_loader):  # tqdm
            inputs = inputs.float().to(device) # [64, 3, 32, 32]
            targets = targets.long().to(device)
            encoded, decoded = model(inputs) # 64*512/784
            # print(inputs.shape, decoded.shape)
            
            MSE_loss = (inputs.reshape(len(targets),-1) - decoded) ** 2
            MSE_loss = args.mse_weight * MSE_loss.view(1, -1).sum(1) / len(targets)
            CE_loss = nn.CrossEntropyLoss()(model.fc(encoded)/args.temp, targets)
            # loss = MSE_loss + CE_loss
            # encoded_half, _ = torch.topk(encoded, 100, largest=False)
            # rho_hat = torch.sum(encoded_half, dim=0, keepdim=True)
            rho_hat = torch.sum(encoded, dim=0, keepdim=True)
            Sparse_loss= args.kl_weight * KL_divergence(rho, rho_hat)
            loss = MSE_loss + CE_loss + Sparse_loss # 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(model.fc(encoded), dim=1)
            train_loss += loss.item()
            train_total += len(targets)
            train_correct += predicted.to(device).eq(targets).sum().item()

        train_loss = train_loss / len(train_loader)  # / 938 = 60032 / 64
        train_acc =  train_correct / train_total  # / 60,000
        loss_plot.append(train_loss)
        acc_plot.append(train_acc)
            
        scheduler.step()

        model.eval()
        with torch.no_grad():
            # for inputs, targets in tqdm(val_loader):
            for _, (inputs, targets) in enumerate(val_loader):   # tqdm
                inputs = inputs.float().to(device)
                targets = targets.long().to(device)
                encoded, decoded = model(inputs)
            
                MSE_loss = (inputs.reshape(len(targets),-1) - decoded) ** 2
                MSE_loss = args.mse_weight * MSE_loss.view(1, -1).sum(1) / len(targets)
                CE_loss = nn.CrossEntropyLoss()(model.fc(encoded)/args.temp, targets)
                # loss = MSE_loss + CE_loss
                # encoded_half, _ = torch.topk(encoded, 100, largest=False)
                # rho_hat = torch.sum(encoded_half, dim=0, keepdim=True)
                rho_hat = torch.sum(encoded, dim=0, keepdim=True)
                Sparse_loss= args.kl_weight * KL_divergence(rho, rho_hat)
                loss = MSE_loss + CE_loss + Sparse_loss #  
                
                _, predicted = torch.max(model.fc(encoded), dim=1)
                val_loss = val_loss + loss.item()
                val_total += len(targets)
                val_correct += predicted.to(device).eq(targets).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
        loss_plot1.append(val_loss)
        acc_plot1.append(val_acc)

        if (epoch % 10 == 0) or (val_acc >= temp_acc): # (val_acc >= temp_acc): val_loss < temp_loss)
            # temp_loss = val_loss
            temp_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/%s/scpn_%s_ld_%d_ep_%d_va_%.4f.pth'
                       % (args.data_name, args.data_name, args.latent_dim, epoch, val_acc))

        print(
            'Epoch : %03d  Train Loss: %.3f | Train Acc: %.3f%% | Val Loss: %.3f | Val Acc: %.3f%%'
            % (epoch, train_loss, 100 * train_acc, val_loss, 100 * val_acc))
            # 'Epoch : %03d  Train Loss: %.3f | Val Loss: %.3f ' % (epoch, train_loss,  val_loss))
        

    plt.plot([e for e in range(len(loss_plot))],loss_plot)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('loss changed owing to epoch')
    plt.show()
    plt.savefig('%s train loss.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    plt.close()
    
    plt.plot([e for e in range(len(loss_plot1))],loss_plot1)
    plt.xlabel('epoch')
    plt.ylabel('val loss')
    plt.title('loss changed owing to epoch')
    plt.show()
    plt.savefig('%s val loss.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    plt.close()
    
    plt.plot([e for e in range(len(acc_plot))],acc_plot)
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    plt.title('accuracy changed owing to epoch')
    plt.show()
    plt.savefig('%s train acc.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    plt.close()
    
    plt.plot([e for e in range(len(acc_plot1))],acc_plot1)
    plt.xlabel('epoch')
    plt.ylabel('val accuracy')
    plt.title('accuracy changed owing to epoch')
    plt.show()
    plt.savefig('%s val acc.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    plt.close()


if __name__ == '__main__':

    main() 
    # for i in range(10):
        # main(numclass=i)
