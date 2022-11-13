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
    if (net_type == 'Blenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'Balexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'B3conv3fc'):
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
    
def BPGD(model, criterion, len, image,labels, eps=0.3, alpha=2/255, iters=40,num_ens=1) :
    # images = images.to(device)
    # labels = labels.to(device)
    # loss = nn.CrossEntropyLoss()
        
    # ori_images = images.data
    training_loss = 0.0
    accs = []
    kl_list = []
    for i in range(iters) :  
          inputs= image.to(device)
          labels=labels.to(device)
          outputs = torch.zeros(inputs.shape[0], model.num_classes, num_ens).to(device)
          inputs.requires_grad = True
          kl = 0.0
          for j in range(num_ens):
              net_out, _kl = model(inputs)
              kl += _kl
              outputs[:, :, j] = F.log_softmax(net_out, dim=1)
          
          kl = kl / num_ens
          kl_list.append(kl.item())
          log_outputs = utils.logmeanexp(outputs, dim=2)

          beta = 1/ len
          loss = criterion(log_outputs, labels, kl, beta)
          loss.backward()
          adv_images = inputs + alpha*inputs.grad.sign()
          eta = torch.clamp(adv_images - inputs, min=-eps, max=eps)
          images = torch.clamp(inputs + eta, min=0, max=1).detach_()
    return images

def test_attack_PGD_freq(freq,dataset,test_loader,inputs,outputs):
    # n_epochs = cfg.n_epochs
    lr = cfg.lr

    criterion = nn.CrossEntropyLoss()
    dict={}
    pgdstep=[]
    accuracies=[]
    for model in freq:
        fmodel = getFModel(model, inputs, outputs).to(device)
        ckpt_name = f'checkpoints/{dataset}/frequentist/model_{fmodel}.pt'
        model.load_state_dict(torch.load(ckpt_name))
        model = model.eval().cuda()
        test_loss, test_acc = test_model(model, criterion, test_loader)
        for step in range(0,100,10):
            atk = PGD(model, eps=0.03, alpha=lr, steps=step)
            atk.set_return_type('int') # Save as integer.
            atk.save(data_loader=test_loader, save_path="adv_data/{dataset}_{step}_PGD.pt", verbose=True)
            # test_loss = test_loss/len(test_loader.dataset)
            
            adv_images, adv_labels = torch.load("adv_data/{dataset}_{step}_PGD.pt")
            adv_data = TensorDataset(adv_images.float()/255, adv_labels)
            adv_loader = DataLoader(adv_data, batch_size=cfg.batch_size, shuffle=False)
            adv_loss, adv_acc = test_model(model, criterion, adv_loader)
            pgdstep.append(step)
            accuracies.append(adv_acc)
        dict[model]={"accu":accuracies, 'steps':pgdstep}
    return dict       
def test_attack_PGD_bayes(bay,attack,dataset,test_loader,inputs,outputs):
    layer_type = cfg2.layer_type
    activation_type = cfg2.activation_type
    priors = cfg2.priors
    criterion = metrics.ELBO(len(test_loader)).to(device)
    
    train_ens = cfg2.train_ens
    valid_ens = cfg2.valid_ens
    n_epochs = cfg2.n_epochs
    lr_start = cfg2.lr_start
    
    # valid_size = cfg2.valid_size
    # batch_size = cfg2.batch_size
    
    # trainset, testset, inputs, outputs = data.getDataset(dataset)
    # train_loader, valid_loader, test_loader = data.getDataloader(
    #         trainset, testset, valid_size, batch_size, num_workers)
    dict={}
    dict_step={}
    for model in bay:
        model2 = getModel(model, inputs, outputs, priors, layer_type, activation_type).to(device)
    
        ckpt_dir = f'checkpoints/{dataset}/bayesian'
        ckpt_name = f'checkpoints/{dataset}/bayesian/model_{model}_{layer_type}_{activation_type}.pt'
    
        model2.load_state_dict(torch.load(ckpt_name))
        model2 = model2.eval().cuda()
        dict={}
        pgdstep=[]
        accuracies=[]
        for step in in range(0,100,10):
            
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                model2.train()
                btest_loss=0.0
                btest_accs=[]
                adv_images = BPGD(model2,criterion,len(test_loader),images,labels, eps=0.03, alpha=cfg.lr, iters=step)
                labels = labels.to(device)
                outputs = torch.zeros(adv_images.shape[0], model2.num_classes, 1).to(device)
                kl = 0.0
                for j in range(1):
                    net_out, _kl = model2(adv_images)
                    kl += _kl
                    outputs[:, :, j] = F.log_softmax(net_out, dim=1).data
            
                log_outputs = utils.logmeanexp(outputs, dim=2)
            
                criterion = metrics.ELBO(len(test_loader)).to(device)
                
                beta = 1/ len(test_loader)
                btest_loss += criterion(log_outputs, labels, kl, beta).item()
                btest_accs.append(metrics.acc(log_outputs, labels))
            
            pgdstep.append(step)
            accuracies.append(np.mean(btest_accs))
        dict[model]={"accu":accuracies, 'steps':pgdstep}
    return dict       

def plot_PGD(dic_1,dic_2):
    
    fmodels= dic_1.keys() 
    bmodels= dic_2.keys()
    for f in fmodels:
        dic=dic1[f]
        x=dic['steps']
        y=dic['accu']
        plt.plot(x, y, label=f, marker='o')
        
    for b in fmodels:
        dic=dic2[b]
        x=dic['steps']
        y=dic['accu']
        plt.plot(x, y, label=b, marker='v')

    plt.title('Training and Validation accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Test Accuracy')
    plt.savefig('att_figure/PGD.png')
        
def test(dataset,attack):
    freq=[]
    bay=[]
    for model in ['alexnet','lenet','3conv3fc']:
        fckpt_name = f'checkpoints/{dataset}/frequentist/model_{model}.pt'
        bckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'
        if os.path.exists(fckpt_name):
            freq.append(model)
        if os.path.exists(bckpt_name):
            bay.append('B'+model)
    
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    
    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(trainset, testset, valid_size, batch_size, num_workers)
    
    if attack=='PGD':
        dict_1=test_attack_PGD_freq(freq,attack,dataset,test_loader,inputs,outputs)
        dict_2=test_attack_PGD_bayes(bay,attack,dataset,test_loader,inputs,outputs)
        plot_PGD(dict_1,dict_2)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    # parser.add_argument('--method', default='frequentist', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10]')
    parser.add_argument('--attack',default='PGD', type=str, help='attack = [PGD/FGSM]')
    args = parser.parse_args()
    
    test(args.dataset,args.attack)
