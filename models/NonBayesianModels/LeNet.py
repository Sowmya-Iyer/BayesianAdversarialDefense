import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self, num_classes, inputs=3,activation_type=None):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        if activation_type=='softplus':
            self.act = F.softplus
        elif activation_type=='relu' or activation_type==None:
            self.act = F.relu
        else:
            raise ValueError("Only softplus or relu supported")

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.act(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.act(self.fc1(out))
        out = self.act(self.fc2(out))
        out = self.fc3(out)

        return(out)
