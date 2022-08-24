import torch

import torch.nn as nn

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                


class DeepLabv3_plus(nn.Module):
    def __init__(self, os=16, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Output stride: {}".format(os))
         
        super(DeepLabv3_plus, self).__init__()

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

# class DeepLabv3_plus(nn.Module):
#     def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True):
#         if _print:
#             print("Constructing DeepLabv3+ model...")
#             print("Number of classes: {}".format(n_classes))
#             print("Output stride: {}".format(os))
#             print("Number of Input Channels: {}".format(nInputChannels))
#         super(DeepLabv3_plus, self).__init__()

#         # Atrous Conv
#         self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)

#         # ASPP
#         if os == 16:
#             rates = [1, 6, 12, 18]
#         elif os == 8:
#             rates = [1, 12, 24, 36]
#         else:
#             raise NotImplementedError

#         self.aspp1 = ASPP_module(2048, 256, rate=rates[0])
#         self.aspp2 = ASPP_module(2048, 256, rate=rates[1])
#         self.aspp3 = ASPP_module(2048, 256, rate=rates[2])
#         self.aspp4 = ASPP_module(2048, 256, rate=rates[3])

#         self.relu = nn.ReLU()

#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(2048, 256, 1, stride=1, bias=False),
#                                              nn.BatchNorm2d(256),
#                                              nn.ReLU())

#         self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(256)

#         # adopt [1x1, 48] for channel reduction.
#         self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(48)

#         self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.BatchNorm2d(256),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.BatchNorm2d(256),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

#     def forward(self, input):
#         x, low_level_features = self.resnet_features(input)
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
#                                 int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

#         low_level_features = self.conv2(low_level_features)
#         low_level_features = self.bn2(low_level_features)
#         low_level_features = self.relu(low_level_features)


#         x = torch.cat((x, low_level_features), dim=1)
#         x = self.last_conv(x)
#         x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

#         return x

#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def __init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 # m.weight.data.normal_(0, math.sqrt(2. / n))
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

# def get_1x_lr_params(model):
#     """
#     This generator returns all the parameters of the net except for
#     the last classification layer. Note that for each batchnorm layer,
#     requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
#     any batchnorm parameter
#     """
#     b = [model.resnet_features]
#     for i in range(len(b)):
#         for k in b[i].parameters():
#             if k.requires_grad:
#                 yield k


# def get_10x_lr_params(model):
#     """
#     This generator returns all the parameters for the last layer of the net,
#     which does the classification of pixel into classes
#     """
#     b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
#     for j in range(len(b)):
#         for k in b[j].parameters():
#             if k.requires_grad:
#                 yield k