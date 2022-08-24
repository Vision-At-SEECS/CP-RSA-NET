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
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         ch_in_g = 16
#         print(out_planes, out_planes//16)
#         self.bn = nn.GroupNorm(out_planes//16, out_planes) if bn else None
        self.relu = nn.ReLU() if relu else nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3, gamma=2, b=1, pool_types=['avg', 'max']):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((math.log(channel, 2) + b)/ gamma))
        k_size = t if t % 2 else t + 1
#         k_size = 3
#         print('k_size is: ', k_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
#         print(x.size())
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

#         print('avg', y.size())
        # Two different branches of ECA module
#         print('y.squeeze, y.squeeze.transpose: ', x.size(), y.size(), y.squeeze(-1).size(), y.squeeze(-1).transpose(-1, -2).size(), )
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         print('eca conv', y.size())
        ch_att_sum = y
        
        # feature descriptor on the global spatial information
        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        
        y = ch_att_sum + y
        
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)    
    
        
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 print(avg_pool.size())
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


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

#         return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        return torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)  
class SpatialGate(nn.Module):
    def __init__(self, channels):
        super(SpatialGate, self).__init__()
        kernel_size = 7 #original
        kernel_size = 5
#         gamma=2
#         b=1
        
#         t = int(abs((math.log(channels, 2) + b)/ gamma))
#         kernel_size = t if t % 2 else t + 1
        print(channels, kernel_size, 'cbam\'s kernel size')
        self.compress = ChannelPool()
        
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress1, x_compress2 = self.compress(x)
        x_out = self.spatial(x_compress1) + self.spatial(x_compress2)
        
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class ChannelPool2(nn.Module):
    def forward(self, x):
#         print('torch.max, torch.max.unsqueeze: ', x.size(), torch.max(x,2)[0].size(), torch.max(x,1)[0].unsqueeze(1).size())
        k = 3
        return torch.cat( (\
                           torch.max(\
                                     F.max_pool2d(x, k, stride=1, padding=(k-1)//2)\
                                     ,1)[0].unsqueeze(1)\
                           ,\
                           torch.mean(\
                                      F.avg_pool2d(x, k, stride=1, padding=(k-1)//2)\
                                      ,1).unsqueeze(1))\
                         ,\
                         dim=1\
                        )

class SpatialGate2(nn.Module):
    def __init__(self, channels):
        super(SpatialGate2, self).__init__()
        kernel_size = 7 #original
#         kernel_size = 5
        gamma=2
        b=1
        
        t = int(abs((math.log(channels, 2) + b)/ gamma))
        kernel_size = t if t % 2 else t + 1
#         print(channels, kernel_size, 'cbam\'s kernel size')
        kernel_size = 3
#         self.spatial2_1 = BasicConv(channels, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#         kernel_size = 5
# #         self.spatial2_2 = BasicConv(channels, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#         kernel_size= 7
        self.compress = ChannelPool2()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

    

class ASPP_SpatialGate(nn.Module):
    def __init__(self, in_planes, out_planes=None, rates=[1, 2, 3]):
        super(ASPP_SpatialGate, self).__init__()

        
#         self.aspp1 = ASPP_module(in_planes, in_planes//2, rate=rates[0])
#         self.aspp2 = ASPP_module(in_planes, in_planes//2, rate=rates[1])
#         self.aspp3 = ASPP_module(in_planes, in_planes//2, rate=rates[2])
        kernel_size = 3
        
        self.aspp1 = nn.Conv2d(in_planes, in_planes//2, kernel_size=kernel_size, stride=1, padding=rates[0], dilation=rates[0], bias=False)
        self.aspp2 = nn.Conv2d(in_planes, in_planes//2, kernel_size=kernel_size, stride=1, padding=rates[1], dilation=rates[1], bias=False)
        self.aspp3 = nn.Conv2d(in_planes, in_planes//2, kernel_size=kernel_size, stride=1, padding=rates[2], dilation=rates[2], bias=False)
        
        
        
        self.compress = ChannelPool()
        
        kernel_size = 3
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#         self.spatial_2x3 = BasicConv(4, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress1 = self.compress(self.aspp1(x))
        
        x_compress2 = self.compress(torch.cat((self.aspp2(x), self.aspp3(x)), dim=1))
        
#         print(x_compress1.size(), x_compress2.size())
        x_compress = x_compress1 + x_compress2
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


    

class ASPP_SpatialGate2(nn.Module):
    def __init__(self, in_planes, out_planes=None, rates=[1, 2, 3]):
        super(ASPP_SpatialGate2, self).__init__()

        kernel_size = 3
        
#         self.aspp1 = nn.Conv2d(in_planes, in_planes//2, kernel_size=kernel_size, stride=1, padding=rates[0], dilation=rates[0], bias=False)
        self.aspp2 = nn.Conv2d(in_planes, 1, kernel_size=kernel_size, stride=1, padding=rates[0], dilation=rates[0], bias=False)
        self.aspp3 = nn.Conv2d(in_planes, 1, kernel_size=kernel_size, stride=1, padding=rates[1], dilation=rates[1], bias=False)
        
        
        self.compress = ChannelPool()
        
        kernel_size = 3
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
#         self.spatial_2x3 = BasicConv(4, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)


    def forward(self, x):
#         x_compress1 = self.compress(self.aspp1(x))
        x_compress1 = self.compress(x)
        x_compress2 = torch.cat((self.aspp2(x), self.aspp3(x)), dim=1)
#         x_compress2 = self.compress(torch.cat((self.aspp2(x), self.aspp3(x)), dim=1))
        
#         print(x_compress1.size(), x_compress2.size())
        x_compress = x_compress1 + x_compress2
#         x_compress = torch.cat((x_compress1, x_compress2), dim=1) #concatenate instead of sum, here input channels will be 4 or 6 instead of 2
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

    
    
    
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         torch.cuda.empty_cache()
        self.ChannelGate = eca_layer(gate_channels, pool_types = pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(gate_channels)
#             self.SpatialGate = SpatialGate2(gate_channels)
#             self.SpatialGate = ASPP_SpatialGate(gate_channels, rates=[1, 2, 3])
#             self.SpatialGate = ASPP_SpatialGate2(gate_channels, rates=[2, 3])

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
