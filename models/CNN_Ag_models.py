import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F

import math
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

from torch.autograd import Function
from typing import Any, Optional, Tuple

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


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d_(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

# https://blog.csdn.net/baidu_36161077/article/details/81388141
class Conv1d_(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d_, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# custom conv1d, because pytorch don't have "padding='same'" option.

def conv1d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    if input_rows % 2 == 0:
        out_rows = (input_rows + stride[0] - 1) // stride[0]
    else:
        out_rows = (input_rows) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
   # padding_cols = max(0, (out_rows - 1) * stride[0] +
                       # (filter_rows - 1) * dilation[0] + 1 - input_rows)
    # padding_cols = 0
    # cols_odd = (padding_rows % 2 != 0)
    if rows_odd:
        input = pad(input, [0, int(rows_odd)])
    return F.conv1d(input, weight, bias, stride,
                  padding=(padding_rows) // 2,
                  dilation=dilation, groups=groups)


# pure resnet


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv16x16(in_planes, planes, downsample_size):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, planes, kernel_size=16, stride=downsample_size)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Block_Ag(nn.Module):
    def __init__(self, inplanes, planes, block_index, downsample_size,  groups=1,
                 norm_layer=None):
        super(Block_Ag, self).__init__()  
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.block_index = block_index
        self.maxpooling = nn.MaxPool1d(kernel_size=downsample_size)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv1d_(inplanes, planes, kernel_size=16, stride=downsample_size) # change according to the leads of the input ECG data
        self.bn2 = norm_layer(planes)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = Conv1d_(planes, planes, kernel_size=16, stride=1)
    def forward(self, x):
        identity = x
        #print(x.shape)
        identity = self.maxpooling(identity)
        #print(identity.shape)
        if self.block_index % 4 == 0 and self.block_index > 0:
            y = torch.zeros_like(identity)
            identity = torch.cat([identity,y],1)
        if not self.block_index == 0:  
            x = self.bn1(x)
            x = self.relu(x)
        out = self.conv1(x)
        #print(out.shape)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        #print(out.shape)
        out += identity
        #if self.block_index%4 == 0:
        #    print(out.shape)
        return out

class CNN_Ag(nn.Module):
    def __init__(self, block, input_channels=12, num_classes=9, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,DG_method=None,domain_classes=1):
        super(CNN_Ag,self).__init__()
        self.DG_method=DG_method
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv1d_(input_channels, self.inplanes, kernel_size=16, stride=1, 
                               bias=False) # change according to the leads of the input ECG data
        
        self.res_layers = self._make_layer(block)
        self.bn = norm_layer(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
        self.fc_1_domain = nn.Linear(256, 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.ReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        
        self.grl = GRL()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, 0, 0.05)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block):
      
        layers = []
        conv_subsample_lengths = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2]
        #conv_subsample_lengths = 
        for index, subsample_length in enumerate(conv_subsample_lengths):
            num_filters = 2**int(index/4) * 32
            layer = block(self.inplanes, num_filters, index, subsample_length)
            layers.append(layer)
            self.inplanes = num_filters
        return nn.Sequential(*layers)
    def forward(self, x):
        features = []
        x = x.transpose(1,2)
        x = self.conv1(x)
        #print(x.shape)
        for i in range(16):
            x = self.res_layers[i](x)
            if i%4 == 0:
                features.append(x)
        #x = self.res_layers(x)
        #print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        
        # x is feature
        
        if self.DG_method=='MMD':
            x_cls=self.fc(x)
            return x,x_cls
            
            
            
        
        if self.DG_method=='DG_GR':
            
            y=self.grl(x)
            y=self.fc_1_domain(y)  # 64
            y=self.dp_domain(y)
            y=self.relu_domain(y)
            y=self.fc_2_domain(y)  # 3
            
            x=self.fc(x)
            
            return x,y
        
        else:
        
            x = self.fc(x)

            return x
            

def cnn_Ag(pretrained=False, progress=True, **kwargs):
    model = CNN_Ag(Block_Ag, **kwargs)
    return model


# if __name__ == '__main__':
#     model = cnn_Ag(input_channels=12, num_classes=4)
#     x = torch.Tensor(32,12,1000)
#     logit_s = model(x) 
#     print(logit_s.shape)
