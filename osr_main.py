import torch
import argparse
import shutil
from tqdm import tqdm
from tensorboardX import SummaryWriter
from xmlrpc.client import Boolean
import matplotlib.pyplot as plt
from loss import *
from model import SparseAutoencoder_all
from osr_data import *

parser = argparse.ArgumentParser(description='Train Convolutionary Prototype Learning Models')

# dataset
# parser.add_argument('--data_name', default='mnist', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=1, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='cifar10', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

parser.add_argument('--data_name', default='svhn', type=str, help='dataset name to use')
parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
parser.add_argument('--num_classes', default=10, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='cifar100', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=100, type=int, help='class number for the dataset')

# parser.add_argument('--data_name', default='tiny_imagenet', type=str, help='dataset name to use')
# parser.add_argument('--data_channel', default=3, type=int, help='channel of dataset')
# parser.add_argument('--num_classes', default=200, type=int, help='class number for the dataset')

parser.add_argument('--feature_dim', default=512, type=int, help='feature dimension of original data')
parser.add_argument('--latent_dim', default=20, type=int, help='latent dimension of autoencoder feature')

# model
parser.add_argument('--epochs', default=300, type=int, help='total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size') 
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--cls_weight', type=float, default=1e-1, help='ce learning weight')
parser.add_argument('--pl_weight', type=float, default=1e-3, help='pl learning weight')
parser.add_argument('--kl_weight', type=float, default=1e-1, help='kl divergence weight')
parser.add_argument('--mse_weight', type=float, default=1e-2, help='mse learning weight')

# optimization
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-10, help='weight decay') 
parser.add_argument('--gamma', type=float, default=0.1, help='optimization gamma') 

# sparse 
parser.add_argument('--K', default=10, type=int, help='sparse dimension of latent feature')
# parser.add_argument('--use_sparse', type=Boolean, default=True, help='sparse autoencoder')
parser.add_argument('--use_sparse', type=Boolean, default=False, help='sparse autoencoder')
parser.add_argument('--L1', type=Boolean, default=True, help='L1')
# parser.add_argument('--L1', type=Boolean, default=False, help='L2')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if args.data_name=='mnist':
    train_loader, val_loader, _= mnist_data_loader(batch_size=args.batch_size)
elif args.data_name=='cifar10':
    train_loader, val_loader, _ = cifar10_data_loader(batch_size=args.batch_size)
elif args.data_name=='cifar100':
    train_loader, val_loader, _ = cifar100_data_loader(batch_size=args.batch_size)
elif args.data_name=='svhn':
    train_loader, val_loader, _ = svhn_data_loader(batch_size=args.batch_size)
elif args.data_name=='tiny_imagenet':
    train_loader, val_loader, _ = imagenet_data_loader(batch_size=args.batch_size)


def main(): # numclass=0
    print(args)

    model = SparseAutoencoder_all(in_channel=args.data_channel,num_classes=args.num_classes,feature_dim=args.feature_dim, latent_dim=args.latent_dim).to(device) #
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)   #
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 210, 270], gamma=args.gamma)
    rho = torch.FloatTensor([0.005 for _ in range(args.K)]).unsqueeze(0).to(device) # args.latent_dim

    cls = CLoss()
    sls = SparseLoss()
    mse = MSELoss()
    temp_acc = 0.93
    # mse = torch.nn.MSELoss()
    loss_plot, loss_plot1, acc_plot, acc_plot1, mse_plot, mse_plot1 = [], [], [], [], [], []
    for epoch in range(args.epochs):
        train_mse_loss, train_loss, train_correct, train_total = 0, 0, 0, 0
        val_mse_loss, val_loss, val_correct, val_total = 0, 0, 0, 0

        model.train()
        # for inputs, targets in tqdm(train_loader):
        for _, (inputs, targets) in enumerate(train_loader):  # tqdm
            inputs = inputs.float().to(device) # [64, 3, 32, 32]
            targets = targets.long().to(device)

            feat, encoded, decoded = model(inputs)

            MSE_loss = args.mse_weight * mse(feat,decoded,targets)
            C_loss = args.cls_weight * cls(feat, decoded, targets, args.temp, args.L1)
            if args.use_sparse:
                Sparse_loss = args.kl_weight * sls(rho, encoded, targets, args.K)
                loss = C_loss + Sparse_loss
            else: 
                loss = C_loss
            # print(MSE_loss, C_loss, Sparse_loss)
            # loss = C_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicts = torch.min(L2_dist(feat, decoded), dim=1)
            train_loss += loss.item()
            train_total += len(targets)
            train_correct += predicts.to(device).eq(targets).sum().item()
            train_mse_loss += MSE_loss.item()
            
        train_loss = train_loss / len(train_loader)  # / 938 = 60032 / 64
        train_acc =  train_correct / train_total  # / 60,000
        train_mse_loss = train_mse_loss / len(train_loader)  # / 938 = 60032 / 64
        loss_plot.append(train_loss)
        acc_plot.append(train_acc)
        mse_plot.append(train_mse_loss)
            
        scheduler.step()

        model.eval()
        with torch.no_grad():
            # for inputs, targets in tqdm(val_loader):
            for _, (inputs, targets) in enumerate(val_loader):  
                inputs = inputs.float().to(device) 
                targets = targets.long().to(device)
                feat, encoded, decoded = model(inputs)

                MSE_loss = args.mse_weight * mse(feat,decoded,targets)
                C_loss = args.cls_weight * cls(feat, decoded, targets, args.temp, args.L1)
                if args.use_sparse:
                    Sparse_loss = args.kl_weight * sls(rho, encoded, targets, args.K)
                    loss = C_loss + Sparse_loss
                else: 
                    loss = C_loss
                # loss = C_loss
                
                _, predicts = torch.min(L2_dist(feat, decoded), dim=1)
                val_loss = val_loss + loss.item()
                val_total += len(targets)
                val_correct += predicts.to(device).eq(targets).sum().item()
                val_mse_loss += MSE_loss.item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            val_mse_loss = val_mse_loss / len(val_loader) 
        loss_plot1.append(val_loss)
        acc_plot1.append(val_acc)
        mse_plot1.append(val_mse_loss)
        
        # save results
        if (val_acc >= temp_acc):
            temp_acc = val_acc
            torch.save(model.state_dict(), 'osr_saved_models/%s/osr_%s_ld_%d_ep_%d_L1_%d_s_%d_%.4f.pth'
                       % (args.data_name, args.data_name, args.latent_dim, epoch, args.L1, args.use_sparse, val_acc))
        # print results
        print(
            'Epoch : %03d  Train Loss: %.3f | Train Acc: %.3f%% | Val Loss: %.3f | Val Acc: %.3f%%'
            % (epoch, train_loss, 100 * train_acc, val_loss, 100 * val_acc))


        # record training logs
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar("train/mse_loss", train_mse_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar("val/mse_loss", val_mse_loss, epoch)


    
    # # plot loss/acc/mse curves
    # plt.plot([e for e in range(len(loss_plot))],loss_plot)
    # plt.xlabel('epoch')
    # plt.ylabel('train loss')
    # plt.title('loss changed owing to epoch')
    # plt.show()
    # plt.savefig('%s_train_loss.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    # plt.close()
    
    # plt.plot([e for e in range(len(loss_plot1))],loss_plot1)
    # plt.xlabel('epoch')
    # plt.ylabel('val loss')
    # plt.title('loss changed owing to epoch')
    # plt.show()
    # plt.savefig('%s_val_loss.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    # plt.close()
    
    # plt.plot([e for e in range(len(acc_plot))],acc_plot)
    # plt.xlabel('epoch')
    # plt.ylabel('train accuracy')
    # plt.title('accuracy changed owing to epoch')
    # plt.show()
    # plt.savefig('%s_train_acc.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    # plt.close()
    
    # plt.plot([e for e in range(len(acc_plot1))],acc_plot1)
    # plt.xlabel('epoch')
    # plt.ylabel('val accuracy')
    # plt.title('accuracy changed owing to epoch')
    # plt.show()
    # plt.savefig('%s_val_acc.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    # plt.close()
    
    # plt.plot([e for e in range(len(mse_plot))],mse_plot)
    # plt.xlabel('epoch')
    # plt.ylabel('train mse')
    # plt.title('mse changed owing to epoch')
    # plt.show()
    # plt.savefig('%s_train_mse.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    # plt.close()
    
    # plt.plot([e for e in range(len(mse_plot1))],mse_plot1)
    # plt.xlabel('epoch')
    # plt.ylabel('val mse')
    # plt.title('mse changed owing to epoch')
    # plt.show()
    # plt.savefig('%s_val_mse.png'%(args.data_name), bbox_inches='tight', pad_inches=.05) 
    # plt.close()


if __name__ == '__main__':
    
    # remove old log file
    logdir = './tensorboard/GCPL/'
    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir)

    main()
