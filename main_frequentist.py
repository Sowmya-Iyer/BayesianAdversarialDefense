from __future__ import print_function

import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler,SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

import data
import utils
import metrics
import config_frequentist as cfg
from models.NonBayesianModels.AlexNet import AlexNet
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC
def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter
import numpy as np


def getModel(net_type, inputs, outputs,activation):
    if (net_type == 'lenet'):
        return LeNet(outputs, inputs,activation)
    elif (net_type == 'alexnet'):
        return AlexNet(outputs, inputs,activation)
    elif (net_type == '3conv3fc'):
        return ThreeConvThreeFC(outputs, inputs,activation)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(model, optimizer, criterion, trainloader):
    model.train()
    # print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # epoch_acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        # train_running_correct.append(epoch_acc) 
    epoch_loss = train_running_loss / counter
    # acc = torch.mean(train_running_correct)
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# def train_model(net, optimizer, criterion, train_loader):
#     train_loss = 0.0
#     net.train()
#     accs = []
#     for data, target in train_loader:
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = net(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()*data.size(0)
#         accs.append(metrics.acc(output.detach(), target))
#     return train_loss, np.mean(accs)

def validate_model(model, criterion, testloader):
    model.eval()
    # we need two lists to keep track of class-wise accuracy
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # calculate the accuracy for each class
            # correct  = (preds == labels).squeeze()
            # for i in range(len(preds)):
            #     label = labels[i]
            #     class_correct[label] += correct[i].item()
            #     class_total[label] += 1
            # epoch_acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
            # valid_running_correct
    epoch_loss = valid_running_loss / counter
    # acc = torch.mean(valid_running_correct)
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# def validate_model(net, criterion, valid_loader):
#     valid_loss = 0.0
#     net.eval()
#     accs = []
#     for data, target in valid_loader:
#         data, target = data.to(device), target.to(device)
#         output = net(data)
#         loss = criterion(output, target)
#         valid_loss += loss.item()*data.size(0)
#         accs.append(metrics.acc(output.detach(), target))
#     return valid_loss, np.mean(accs)

def run(dataset, net_type):

    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    activation=cfg.activation
    title= f'{net_type}-{dataset}'
    writer = SummaryWriter(title)

    trainset, testset, inputs, outputs = data.getDataset(dataset,net_type)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs,activation).to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    # if os.path.exists(ckpt_name):
    #     net.load_state_dict(torch.load(ckpt_name))
        
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs+1):

        train_loss, train_acc = train_model(net, optimizer, criterion, train_loader)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader)
        lr_sched.step(valid_loss)

        # train_loss = train_loss/len(train_loader.dataset)
        # valid_loss = valid_loss/len(valid_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', train_loss, epoch)
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_min = valid_loss
        
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
