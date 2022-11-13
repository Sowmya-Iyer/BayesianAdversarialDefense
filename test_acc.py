import argparse

import os 
from torch.nn import functional as F

from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.NonBayesianModels.AlexNet import AlexNet
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC

import os.path
import data
import utils
import metrics

import config_bayesian as cfg2
import config_frequentist as cfg

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getBModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')

def getFModel(net_type, inputs, outputs):
    if (net_type == 'lenet'):
        return LeNet(outputs, inputs)
    elif (net_type == 'alexnet'):
        return AlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return ThreeConvThreeFC(outputs, inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')

def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = 1/len(validloader)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)

def test_model(net, criterion, test_loader):
    valid_loss = 0.0
    net.eval()
    accs = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return valid_loss, np.mean(accs)
    
def test_freq(freq,dataset,test_loader,inputs,outputs):
    lr = cfg.lr

    criterion = nn.CrossEntropyLoss()
    dict={}
    
    for model in freq:
        fmodel = getFModel(model, inputs, outputs).to(device)
        ckpt_name = f'checkpoints/{dataset}/frequentist/model_{model}.pt'
        fmodel.load_state_dict(torch.load(ckpt_name))
        fmodel = fmodel.eval().cuda()
        test_loss, test_acc = test_model(fmodel, criterion, test_loader)
        dict[model]={"accu":test_acc, 'model':model}
    return dict       
def test_bayes(bay,dataset,test_loader,inputs,outputs):
    layer_type = cfg2.layer_type
    activation_type = cfg2.activation_type
    priors = cfg2.priors
    criterion = metrics.ELBO(len(test_loader)).to(device)

    train_ens = cfg2.train_ens
    valid_ens = cfg2.valid_ens
    n_epochs = cfg2.n_epochs
    lr_start = cfg2.lr_start
    num_workers = cfg2.num_workers
    valid_size = cfg2.valid_size
    batch_size = cfg2.batch_size
    beta_type = cfg2.beta_type
    
    dict={}
    for model in bay:
        ckpt_name = f'checkpoints/{dataset}/bayesian/model_{model[1:]}_{layer_type}_{activation_type}.pt'
        bmodel = getBModel(model[1:], inputs, outputs, priors, layer_type, activation_type).to(device)
        bmodel.load_state_dict(torch.load(ckpt_name))
        bmodel = bmodel.eval().cuda()
        test_loss, test_acc =validate_model(bmodel, criterion, test_loader, num_ens=valid_ens, beta_type=beta_type) 
        dict[model]={"accu":test_acc, 'model':model}
    return dict
       

def plot_PGD(dic_1,dic_2,dataset):
    fig = plt.figure(figsize = (10, 5))
    fmodels= dic_1.keys() 
    bmodels= dic_2.keys()
    for f in fmodels:
        dic=dic_1[f]
        x=dic['model']
        y=dic['accu']
        plt.bar(x, y, width = 0.4)

    for b in bmodels:
        dic=dic_2[b]
        x=dic['model']
        y=dic['accu']
        plt.bar(x, y, width = 0.4)

    title=f'{dataset} - Test Accuracies'
    save=f'att_figure/{dataset}_Test_Accuracies.png'
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    # plt.show()
    plt.savefig(save)
        
def test(dataset):
    freq=[]
    bay=[]
    layer_type = cfg2.layer_type
    activation_type = cfg2.activation_type
    for model in ['alexnet','lenet','3conv3fc']:
        fckpt_name = f'checkpoints/{dataset}/frequentist/model_{model}.pt'
        bckpt_name = f'checkpoints/{dataset}/bayesian/model_{model}_{layer_type}_{activation_type}.pt'
        if os.path.exists(fckpt_name):
            freq.append(model)
        if os.path.exists(bckpt_name):
            bay.append('B'+model)
    
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    
    trainset, testset, inputs, outputs = data.getDataset(dataset,'alexnet')
    train_loader, valid_loader, test_loader = data.getDataloader(trainset, testset, valid_size, batch_size, num_workers)
    

    dict_1 = dict_2 = {}
    dict_1=test_freq(freq,dataset,test_loader,inputs,outputs)
    dict_2=test_bayes(bay,dataset,test_loader,inputs,outputs)
    plot_PGD(dict_1,dict_2,dataset)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Test Gradient-Based attack")
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10]')
    args = parser.parse_args()
    
    test(args.dataset)

