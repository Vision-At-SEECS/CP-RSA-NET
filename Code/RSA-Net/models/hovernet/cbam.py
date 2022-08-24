import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .deeplabv3plus import ASPP_module

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None #uncomment me for preact
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None # comment me for preact
#         ch_in_g = 16
#         print(out_planes, out_planes//16)
#         self.bn = nn.GroupNorm(out_planes//16, out_planes) if bn else None
        self.relu = nn.ReLU() if relu else nn.SELU()#None

    def forward(self, x):
#         x = self.conv(x) # comment me for preact
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = self.conv(x) #uncomment me for preact
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    

class ChannelGate(nn.Module):
    """Constructs a Channel module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Kernel Size
    """
    def __init__(self, channel, k_size=3, gamma=2, b=1, pool_types=['avg', 'max']):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        k_size =5
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
    

    def forward(self, x):
        
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        ch_att_sum = y

        # feature descriptor on the global spatial information
        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        y = ch_att_sum + y #.unsqueeze(-1)
        
        # Multi-scale information fusion
        y = F.sigmoid(y)
        
        return x * y.expand_as(x)    
    
        
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
#     @torch.jit.script
#     def pointwise(self, x):
#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    def forward(self, x):
#         print('torch.max, torch.max.unsqueeze: ', x.size(), torch.max(x,2)[0].size(), torch.max(x,1)[0].unsqueeze(1).size())

        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
#         return torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)  


class SpatialGate(nn.Module):
    def __init__(self, channels):
        super(SpatialGate, self).__init__()
        kernel_size = 5 #can be changed
        
        print(channels, kernel_size, 'cbam\'s kernel size')
        self.compress = ChannelPool()
        
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=True, bn=True)
#         self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=True, bn=True)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


    
    
    
class SA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=32, pool_types=['avg', 'max'], no_spatial=False):
        super(SA, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, pool_types = pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(gate_channels)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
