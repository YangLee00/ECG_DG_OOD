import torch
import torch.nn as nn
from torchsummary import summary
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

def conv(n_inputs, n_filters, kernel_size=3, stride=1, bias=False) -> torch.nn.Conv1d:
    """Creates a convolution layer for `XResNet`."""
    return nn.Conv1d(n_inputs, n_filters,
                     kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, bias=bias)

def conv_layer(n_inputs: int, n_filters: int,
               kernel_size: int = 3, stride=1,
               zero_batch_norm: bool = False, use_activation: bool = True,
               activation: torch.nn.Module = nn.ReLU(inplace=True)) -> torch.nn.Sequential:
    """Creates a convolution block for `XResNet`."""
    batch_norm = nn.BatchNorm1d(n_filters)
    # initializer batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0. if zero_batch_norm else 1.)
    layers = [conv(n_inputs, n_filters, kernel_size, stride=stride), batch_norm]
    if use_activation: layers.append(activation)
    return nn.Sequential(*layers)

class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""
    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int = 1,
                 activation: torch.nn.Module = nn.ReLU(inplace=True)):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion

        # convolution path
        if expansion == 1:
            layers = [conv_layer(n_inputs, n_hidden, 3, stride=stride),
                      conv_layer(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [conv_layer(n_inputs, n_hidden, 1),
                      conv_layer(n_hidden, n_hidden, 3, stride=stride),
                      conv_layer(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]

        self.convs = nn.Sequential(*layers)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = conv_layer(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool1d(2, ceil_mode=True)

        self.activation = activation

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))

class XResNet(nn.Module):
    def __init__(self, expansion, layers, input_channels=12, num_classes=1000,DG_method=None,domain_classes=1):
        super(XResNet, self).__init__()
        
        self.DG_method=DG_method

        n_filters_stem = [input_channels, 32, 64, 64]
        stem = [conv_layer(n_filters_stem[i], n_filters_stem[i + 1], stride=2 if i == 0 else 1)
                for i in range(3)]

        self.stem = nn.Sequential(*stem)
        n_filters_xres = [64 // expansion, 64, 128, 256, 512]

        res_layers = [self._make_layer(expansion, n_filters_xres[i], n_filters_xres[i + 1],
                                  n_blocks=l, stride=1 if i == 0 else 2)
                        for i, l in enumerate(layers)]

        self.res_layers = nn.Sequential(*res_layers)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters_xres[-1]*expansion, num_classes)
        
        self.fc_1_domain = nn.Linear(n_filters_xres[-1]*expansion, 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.ReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)

        self.grl = GRL()
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, expansion, n_inputs, n_filters, n_blocks, stride):
        return nn.Sequential(
            *[XResNetBlock(expansion, n_inputs if i==0 else n_filters, n_filters, stride if i==0 else 1)
              for i in range(n_blocks)])
    
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.stem(x)
        from IPython import embed
        #embed()
        x = self.maxpool(x)
        x = self.res_layers(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        
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

    def forward(self, x):
        x=x.transpose(1,2)
        return self._forward_impl(x)


def xresnet18 (**kwargs): return XResNet(1, [2, 2,  2, 2], **kwargs)
def xresnet34 (**kwargs): return XResNet(1, [3, 4,  6, 3], **kwargs)
def xresnet50 (**kwargs): return XResNet(4, [3, 4,  6, 3], **kwargs)
def xresnet101(**kwargs): return XResNet(4, [3, 4, 23, 3], **kwargs)
#     model=XResNet(4, [3, 4, 23, 3], input_channels=1, **kwargs)
# #     summary(model,(1,1000))
#     return model
def xresnet152(**kwargs): return XResNet(4, [3, 8, 36, 3], **kwargs)

