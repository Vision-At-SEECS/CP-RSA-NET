import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape

from .cbam import *

####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4
        ch_in_g = 16
        
        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        self.con_map = False
        
        self.skip = True
        self.skip2 = False
        self.skip3 = False
        self.skip_cbam = False

        if self.skip_cbam:
#             self.e0=self.e1=self.e2 = []
#             self.e0 = nn.ModuleList([CBAM(1024), CBAM(1024), CBAM(1024)])
#             self.e1 = nn.ModuleList([CBAM(512), CBAM(512), CBAM(512)])
#             self.e2 = nn.ModuleList([CBAM(256), CBAM(256), CBAM(256)])
#             for i in range(3):
#                 self.e0.append(CBAM(1024))
#                 self.e1.append(CBAM(512))
#                 self.e2.append(CBAM(256))
                
            self.e0 = SA(1024)
            self.e1 = SA(512)
            self.e2 = SA(256)
         
        
        
        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
#             ("gn", nn.GroupNorm(64//ch_in_g, 64)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
#                 ("/gn", nn.GroupNorm(64//ch_in_g, 64)),
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        if self.con_map:
            module_list = [
                    ("bn", nn.BatchNorm2d(66, eps=1e-5)),
#                     ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                    ("relu", nn.ReLU(inplace=True)),
#                     ("conv", nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=True),),
                    ("conv", nn.Conv2d(66, 2, 1, stride=1, padding=0, bias=True),),
                ]
            self.con = nn.Sequential(OrderedDict(module_list))

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        
        if self.skip3:
            self.skip=False
            self.skip2=False
            
            d[-2] = self.e0(d[-2])
            d[-3] = self.e1(d[-3])
            d[-4] = self.e2(d[-4])
            
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])
        del d0, d1, d2, d3
        out_dict = OrderedDict()
        idx = 0 
        for branch_name, branch_desc in self.decoder.items():
             
            if self.skip:
#                     print(d[-2].shape, d[-3].shape, d[-4].shape)
                u3 = self.upsample2x(d[-1]) + self.e0(d[-2])
#                 u3 = self.upsample2x(d[-1]) + self.e0[idx](d[-2])
#                 print(u3.size(), 'u3 before dec')
                u3 = branch_desc[0](u3)
                
                u2 = self.upsample2x(u3) + self.e1(d[-3])
#                 u2 = self.upsample2x(u3) + self.e1[idx](d[-3])
#                 print(u2.size(), 'u2 before dec')
                u2 = branch_desc[1](u2)

                u1 = self.upsample2x(u2) + self.e2(d[-4])
#                 u1 = self.upsample2x(u2) + self.e2[idx](d[-4])
                u1 = branch_desc[2](u1)

                u0 = branch_desc[3](u1)

                out_dict[branch_name] = u0 #[4, 2, 164, 164]
                
                if (self.con_map) & (branch_name=='np'):
                    out_dict["con"] = self.con(torch.cat((u0, u1),dim=1))
#                         out_dict["con"] = self.con(u1)
                    
#                         print(out_dict["con"].size())
            elif self.skip2:
#                     print(d[-2].shape, d[-3].shape, d[-4].shape)
                u3 = self.e0(self.upsample2x(d[-1]) + d[-2])
#                 print(u3.size(), 'u3 before dec')
                u3 = branch_desc[0](u3)
                
                u2 = self.upsample2x(u3) + self.e1(d[-3])
#                 print(u2.size(), 'u2 before dec')
                u2 = branch_desc[1](u2)

                u1 = self.upsample2x(u2) + self.e2(d[-4])
                u1 = branch_desc[2](u1)

                u0 = branch_desc[3](u1)

                out_dict[branch_name] = u0 #[4, 2, 164, 164]
                
            else:
                u3 = self.upsample2x(d[-1]) + d[-2]
                u3 = branch_desc[0](u3)

                u2 = self.upsample2x(u3) + d[-3]
                u2 = branch_desc[1](u2)

                u1 = self.upsample2x(u2) + d[-4]
                u1 = branch_desc[2](u1)

                u0 = branch_desc[3](u1)
                out_dict[branch_name] = u0
                
            idx += 1
        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)
