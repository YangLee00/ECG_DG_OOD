import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

def noop(x): return x

class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else noop

        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())

    def forward(self, x):
        #print("block in",x.size())
        bottled = self.bottleneck(x)  # (, 32, 10000)
        input_1 = [c(bottled) for c in self.convs]#(, 32, 10000)
        input_2 = [self.conv_bottle(x)]#(, 32, 10000)
        out = self.bn_relu(torch.cat(input_1+input_2, dim=1))
        return out

class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn=nn.ReLU(True)
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)

    def forward(self, inp, out): 
        #print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
        #input()
        return self.act_fn(out + self.bn(self.conv(inp)))
        
class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d==0 else n_ks*nb_filters,nb_filters=nb_filters,kss=kss, bottleneck_size=bottleneck_size) for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d==0 else n_ks*nb_filters, n_ks*nb_filters) for d in range(depth//3)])    
        
    def forward(self, x):
        
        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d//3])(input_res, x)
                input_res = x.clone()
        return x

class Inception1d(nn.Module):
    '''inception time architecture'''
    def __init__(self, num_classes=9, input_channels=12, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True,DG_method=None,domain_classes=1):
        super().__init__()
        assert(kernel_size>=40)
        kernel_size = [k-1 if k%2==0 else k for k in [kernel_size,kernel_size//2,kernel_size//4]] #was 39,19,9
        
        layers = [InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)]
        
        self.DG_method=DG_method
       
        n_ks = len(kernel_size) + 1
        #head
        #head = create_head1d(n_ks*nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        #layers.append(head)
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool_final = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2*n_ks*nb_filters, num_classes)  # follow the ori paper!
        
        self.fc_1_domain = nn.Linear(2*n_ks*nb_filters, 64)
        self.fc_2_domain = nn.Linear(64, domain_classes)
        self.relu_domain=nn.ReLU(inplace=True)
        self.dp_domain=nn.Dropout (p=0.5)
        
        self.grl = GRL()


    def forward(self,x):
        x=x.transpose(1,2)
        x = self.layers(x)
        x_avg = self.avgpool(x)
        x_avg = torch.flatten(x_avg, 1)
        x_max = self.maxpool_final(x)
        x_max = torch.flatten(x_max, 1)
        x = torch.cat((x_avg, x_max), 1)
        
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
        depth = self.layers[0].depth
        if(depth>3):
            return ((self.layers[0].im[3:],self.layers[0].sk[1:]),self.layers[-1])
        else:
            return (self.layers[-1])
    
    def get_output_layer(self):
        return self.layers[-1][-1]
    
    def set_output_layer(self,x):
        self.layers[-1][-1] = x
    
def inception1d(**kwargs):
    """Constructs an Inception model
    """
    return Inception1d(**kwargs)


if __name__ =='__main__':
    net = inception1d()
    _input = torch.rand(2, 12, 10000)
    out = net(_input)