from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.BayesianModels.BayesianVGG11 import BBBVGG11
# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'vgg'):
        return BBBVGG11(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader):
    net.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    kl_list = []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        optimizer.zero_grad()

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        # kl = 0.0
        
        net_out, kl = net(inputs)
        kl += _kl
        outputs[:, :, 1] = F.log_softmax(net_out, dim=1)
        
        # kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = 1/len(train_loader)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(log_outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # accs.append(metrics.acc(log_outputs.data, labels))
        # training_loss += loss.cpu().data.numpy()
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc, np.mean(kl_list)

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
            outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
            
            net_out, kl = net(inputs)
            
            outputs[:, :, 1] = F.log_softmax(net_out, dim=1).data

            log_outputs = utils.logmeanexp(outputs, dim=2)

            beta = 1/len(testloader)
            loss = criterion(log_outputs, labels, kl, beta)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(log_outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # calculate the accuracy for each class
            # correct  = (preds == labels).squeeze()
            # for i in range(len(preds)):
            #     label = labels[i]
            #     class_correct[label] += correct[i].item()
            #     class_total[label] += 1
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
    
# def validate_model(net, criterion, validloader):
#     """Calculate ensemble accuracy and NLL Loss"""
#     net.train()
#     valid_loss = 0.0
#     accs = []

#     for i, (inputs, labels) in enumerate(validloader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
#         kl = 0.0
#         for j in range(num_ens):
#             net_out, _kl = net(inputs)
#             kl += _kl
#             outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

#         log_outputs = utils.logmeanexp(outputs, dim=2)

#         beta = 1/len(validloader)
#         valid_loss += criterion(log_outputs, labels, kl, beta).item()
#         accs.append(metrics.acc(log_outputs, labels))

#     return valid_loss/len(validloader), np.mean(accs)


def run(dataset, net_type):

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    # train_ens = cfg.train_ens
    # valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    

    title= f'Bayes-{net_type}-{dataset}'
    writer = SummaryWriter(title)

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_dir = f'ccheckpoints/{dataset}/bayesian'
    ckpt_name = f'ccheckpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    #criterion = metrics.ELBO(len(trainset)).to(device)
    criterion = metrics.ELBO(batch_size).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader)
        lr_sched.step(valid_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', train_loss, epoch)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
