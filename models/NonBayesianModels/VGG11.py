import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)
class VGG11(nn.Module):
    def __init__(self, num_classes, inputs=3,activation_type=None):
        super(VGG11, self).__init__()
        self.in_channels = inputs
        self.num_classes = num_classes
        self.linear_input_size=4*4*3*512
        # convolutional layers 
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu' or activation_type==None:
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            self.act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.act(),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            self.act(),
            nn.Dropout(p=0.5),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # self.act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.act(),
            nn.Dropout(p=0.5),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # self.act(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=self.linear_input_size, out_features=4096),
            self.act(),
            nn.Dropout(p=0.5),
            # nn.Linear(in_features=4096, out_features=4096),
            # self.act(),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        self.linear_input_size = x.view(-1, x.shape[-1]).shape[0]
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        
