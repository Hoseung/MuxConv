import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, 
                        stride=1, 
                        affine=False, 
                        activation=F.relu):
        self.activation = activation

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(
                              in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,    
                              bias=False), 
                            nn.BatchNorm2d(self.expansion * planes, affine=affine))


    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

    def validate(self, x):
        out = self.bn1(self.conv1(x))
        omi1 = out.min()
        oma1 = out.max()
        out = self.activation(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        omi2 = out.min()
        oma2 = out.max()
        out = self.activation(out)
        return out, (min(omi1, omi2), max(oma1, oma2))

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation=F.relu):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.activation = activation

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 
                         stride=1, activation=self.activation)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 
                            stride=2, activation=self.activation)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 
                            stride=2, activation=self.activation)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        self.avgpool = nn.AvgPool2d(8,8)

    def _make_layer(self, block, planes, num_blocks, stride, activation=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        print("before softmax", out)
        return F.log_softmax(out, dim=1)

    def debug(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        print("DEBUG 1", out)
        out = self.layer1(out)
        print("DEBUG 2", out)
        out = self.layer2(out)
        print("DEBUG 3", out)
        out = self.layer3(out)
        print("DEBUG 4", out)
        out = self.avgpool(out)
        print("DEBUG 5", out)
        out = torch.flatten(out, 1)
        print("DEBUG 6", out)
        out = self.linear(out)
        print("DEBUG 7", out)
        print("before softmax", out)
        return F.log_softmax(out, dim=1)

def ResNet20(activation=F.relu):
    return ResNet(BasicBlock, [2, 2, 2], activation=activation)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10, activation=F.relu, affine = True):
        super().__init__()
        self.in_planes = 16
        self.activation = activation
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9, affine=affine)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.1, affine=affine)

        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.basicblock1 = BasicBlock(128, 128, 
                         stride=1, activation=self.activation)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.1, affine=affine)

        self.conv4 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256, momentum=0.1, affine=affine)

        self.basicblock2 = BasicBlock(256, 256, 
                         stride=1, activation=self.activation)

        self.linear = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.minmax=[torch.inf, -torch.inf]


    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.maxpool(out)

        out = self.basicblock1(out)

        out = self.activation(self.bn3(self.conv3(out)))
        out = self.maxpool(out)
        out = self.activation(self.bn4(self.conv4(out)))
        out = self.maxpool(out)

        out = self.basicblock2(out)
        out = self.maxpool(out)

        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        return out #F.log_softmax(out, dim=1)

    def _do_minmax(self, out):
        self.minmax[0] = min(self.minmax[0], out.min())
        self.minmax[1] = max(self.minmax[1], out.max())
        #print("act", out.min(), out.max())

    def validate(self, x):
        """forward + min, max range of activation input
        """
        out = self.bn1(self.conv1(x))
        self._do_minmax(out)
        out = self.activation(out)

        out = self.bn2(self.conv2(out))
        self._do_minmax(out)
        out = self.activation(out)
        out = self.maxpool(out)

        out, mima = self.basicblock1.validate(out)
        self.minmax[0] = min(self.minmax[0], mima[0])
        self.minmax[1] = max(self.minmax[1], mima[1])

        out = self.bn3(self.conv3(out))
        self._do_minmax(out)
        out = self.activation(out)
        out = self.maxpool(out)

        out = self.bn4(self.conv4(out))
        self._do_minmax(out)
        out = self.activation(out)

        out = self.maxpool(out)

        out, mima = self.basicblock2.validate(out)
        out = self.maxpool(out)

        out = torch.flatten(out, 1)
        out = self.linear(out)
        
        #print("activation Input Min, Max")
        #print(self.minmax)
        return out #F.log_softmax(out, dim=1)

    def debug(self, x):
        print("0", x.min(), x.max())
        out = self.conv1(x)
        print("1", out.min(), out.max())
        out = self.bn1(out)
        print("1-1", out.min(), out.max())
        out = self.activation(out)
        print("1-2", out.min(), out.max())
        #print("ZZZZ", out.shape)
        out = self.bn2(self.conv2(out))
        print("2", out.min(), out.max())
        out = self.activation(out)
        print("2-2", out.min(), out.max())
        out = self.maxpool(out)

        out = self.basicblock1(out)
        print("3", out.min(), out.max())

        out = self.activation(self.bn3(self.conv3(out)))
        print("4", out.min(), out.max())
        out = self.maxpool(out)
        out = self.activation(self.bn4(self.conv4(out)))
        print("5", out.min(), out.max())
        out = self.maxpool(out)

        out = self.basicblock2(out)
        print("6", out.min(), out.max())
        out = self.maxpool(out)

        out = torch.flatten(out, 1)
        print("7", out.min(), out.max())
        out = self.linear(out)
        
        return out #F.log_softmax(out, dim=1)

