import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Optional, Tuple

import math


###############################################################################################
# Standard resnet

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
    
def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=[3,3], downsample=None):
        super().__init__()

        if(isinstance(kernel_size,int)): kernel_size = [kernel_size,kernel_size//2+1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes,kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1d(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    '''1d adaptation of the torchvision resnet'''
    def __init__(self, block, layers, kernel_size=5, num_classes=9, input_channels=12, inplanes=128, fix_feature_dim=True, kernel_size_stem = None, stride_stem=2, pooling_stem=True, stride=2,DG_method=None,domain_classes=1):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes

        if(kernel_size_stem is None):
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size,list) else kernel_size
        #stem
        self.conv1 = nn.Conv1d(input_channels, inplanes, kernel_size=kernel_size_stem, stride=stride_stem, padding=(kernel_size_stem-1)//2,bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #backbone
        layers_tmp = []
        for i,l in enumerate(layers):
            if(i==0):
                layers_tmp.append(self._make_layer(block, inplanes, layers[0],kernel_size=kernel_size))
            else:
                layers_tmp.append(self._make_layer(block, inplanes if fix_feature_dim else (2**i)*inplanes, layers[i], stride=stride,kernel_size=kernel_size))
        self.res_layers = nn.Sequential(*layers_tmp)
        #head
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear((inplanes if fix_feature_dim else 2*(2**len(layers)*inplanes)) * block.expansion, num_classes)   # just use avgpooling not follow the ori paper!
        self.fc_1_domain = nn.Linear((inplanes if fix_feature_dim else 2*(2**len(layers)*inplanes)) * block.expansion, 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.ReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        self.grl = GRL()
        self.DG_method=DG_method
       
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,kernel_size=3):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel_size, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def forward(self,x):
        x=x.transpose(1,2)

        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features.append(x)
        x = self.res_layers[0](x)
        features.append(x)
        x = self.res_layers[1](x)
        features.append(x)
        x = self.res_layers[2](x)
        features.append(x)

        x_avg = self.avgpool(x)
        x = torch.flatten(x_avg, 1)
        
        
        
        if self.DG_method=='DG_GR':
            
            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            
            
            x=self.fc(x)
            
            return x,y
        
        if self.DG_method=='MMD':
            x_cls=self.fc(x)
            return x,x_cls
        
        else:
        
            x = self.fc(x)

            return x
    
    
    
    def get_layer_groups(self):
        return (self[6],self[-1])
    
    def get_output_layer(self):
        return self[-1][-1]
        
    def set_output_layer(self,x):
        self[-1][-1]=x



#original used kernel_size_stem = 8
def resnet1d_wang(**kwargs):
    
    if(not("kernel_size" in kwargs.keys())):
        kwargs["kernel_size"]=[5,3]
    if(not("kernel_size_stem" in kwargs.keys())):
        kwargs["kernel_size_stem"]=7
    if(not("stride_stem" in kwargs.keys())):
        kwargs["stride_stem"]=1
    if(not("pooling_stem" in kwargs.keys())):
        kwargs["pooling_stem"]=False
    if(not("inplanes" in kwargs.keys())):
        kwargs["inplanes"]=128


    return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)
