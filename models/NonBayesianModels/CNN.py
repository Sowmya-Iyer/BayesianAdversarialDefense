import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)
class CNN(nn.Module):
    def __init__(self, num_classes, inputs=3,activation_type=None):
        super(CNN, self).__init__()
        self.in_channels = inputs
        self.num_classes = num_classes
        # convolutional layers 
        if activation_type=='softplus':
            self.act = nn.Softplus()
        elif activation_type=='relu' or activation_type==None:
            self.act = nn.ReLU()
        else:
            raise ValueError("Only softplus or relu supported")
        self.network = nn.Sequential(
            
            nn.Conv2d(self.in_channels, 32, kernel_size = 3, padding = 1),
            self.act,
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            self.act,
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            self.act,
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            self.act,
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            self.act,
            nn.Dropout(p=0.4),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            self.act,
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(16384,1024),
            # nn.Dropout(p=0.5),
            # self.act,
            nn.Linear(1024, 512),
            self.act,
            nn.Linear(512, self.num_classes),
        )
    
    def forward(self, xb):
        return self.network(xb)