import torch
from torch import nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl


"""
    # double factorial
    # 3!! = 3 * 1 
    # 4!! = 4 * 2
    # 5!! = 5 * 3 * 1
    # 6!! = 6 * 4 * 2
    # n!! = n * (n-2) * (n-4) * ... * 1 if n is odd
    # n!! = n * (n-2) * (n-4) * ... * 2 if n is even
    # (n-3)!! = 1 or 1*3 or 1*3*5. maybe 1*3. 
"""

class HermitePN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(HermitePN, self).__init__()
        self.num_features = num_features
        
        # Batch normalization layer
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.register_buffer('running_mean2', torch.zeros(num_features))
        self.register_buffer('running_var2', torch.ones(num_features))
    
    def _normalize(self, x):
        if self.training:
            # Calculate mean and variance
            batch_mean = x.mean([0, 2, 3], keepdim=True)
            batch_var = x.var([0, 2, 3], unbiased=False, keepdim=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.view(-1)
        else:
            # Use running statistics
            batch_mean = self.running_mean.view(1, self.num_features, 1, 1)
            batch_var = self.running_var.view(1, self.num_features, 1, 1)
            #print(["VALIDATION"], batch_mean, batch_var)
        
        # Normalize
        x_hat = self._scale(x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat    

    def _normalize2(self, x):
        if self.training:
            # Calculate mean and variance
            batch_mean = x.mean([0, 2, 3], keepdim=True)
            batch_var = x.var([0, 2, 3], unbiased=False, keepdim=True)
            
            # Update running statistics
            self.running_mean2 = (1 - self.momentum) * self.running_mean2 + self.momentum * batch_mean.view(-1)
            self.running_var2 = (1 - self.momentum) * self.running_var2 + self.momentum * batch_var.view(-1)
        else:
            # Use running statistics
            batch_mean = self.running_mean2.view(1, self.num_features, 1, 1)
            batch_var = self.running_var2.view(1, self.num_features, 1, 1)
        
        # Normalize
        x_hat = self._scale(x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat    
        
    def _scale(self, x):
        # weight(gamma)과 bias(beta)는 learnable parameter임. 
        return self.weight.view(1, self.num_features, 1, 1) * x + self.bias.view(1, self.num_features, 1, 1)

    def forward(self, x):
        # Normalized Hermite polynomials
        h0 = 1
        h1 = x
        h2 = (x**2 - 1)/math.sqrt(2)

        # Coefficients of Hermite approximation of ReLU
        f0 = 1/math.sqrt(2*math.pi)
        f1 = 1/2
        f2 = 1/math.sqrt(2*math.pi*2) 
        
        h0_bn = h0 * f0
        h1_bn = self._normalize(f1*h1)
        h2_bn = self._normalize2(f2*h2)

        result = h0_bn + h1_bn + h2_bn
        return result
        

class BasicBlockHer(nn.Module):
    """BasicBlock with Hermitian activation"""
    expansion = 1

    def __init__(self, in_planes, planes, 
                        stride=1, 
                        affine=False):

        super(BasicBlockHer, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes, affine=affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                                nn.Conv2d(
                              in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False)#, 
                            # nn.BatchNorm2d(self.expansion * planes, affine=affine)
            )
        self.herPN1 = HermitePN(planes)
        self.herPN2 = HermitePN(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.herPN1(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.herPN2(out)
        return out


class ResNetHer(pl.LightningModule):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.hermitepn = HermitePN(self.in_planes)
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(BasicBlockHer, 16, num_blocks[0], 
                         stride=1)
        self.layer2 = self._make_layer(BasicBlockHer, 32, num_blocks[1], 
                            stride=2)
        self.layer3 = self._make_layer(BasicBlockHer, 64, num_blocks[2], 
                            stride=2)
        self.linear = nn.Linear(64 * BasicBlockHer.expansion, num_classes)
        
        self.avgpool = nn.AvgPool2d(8,8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.hermitepn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        return F.log_softmax(out, dim=1)