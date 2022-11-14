import torch.nn as nn
import math
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BBBVGG11(ModuleWrapper):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBVGG11, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 64, 3, padding=1, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(64, 128, 3, padding=1, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(128, 256, 3, padding=1, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.conv4 = BBBConv2d(256, 256, 3, padding=1, bias=True, priors=self.priors)
        self.act4 = self.act()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = BBBConv2d(256, 512, 3, padding=1, bias=True, priors=self.priors)
        self.act5 = self.act()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv6 = BBBConv2d(512, 512, 3, padding=1, bias=True, priors=self.priors)
        self.act6 = self.act()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = FlattenLayer(512*7*7)
        self.fc1 = BBBLinear(512*7*7, 4096, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BBBLinear(4096, outputs, bias=True, priors=self.priors)
 