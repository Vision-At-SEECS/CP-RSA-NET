import math
import numpy as np

import torch
import torch.nn.functional as F

from matplotlib import cm

from collections import OrderedDict

# import os
import cv2 
# import glob 
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

from torch.autograd import Variable

class _SimpleSegmentationModel(torch.nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
########################################################################################

def sigmoid_focal_loss(
    targets,
    inputs,
    alpha: float = 0.75,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
#     p = inputs / torch.sum(inputs, -1, keepdim=True)
#     print(p)
#     print('p', np.unique(p.detach().cpu().numpy()), p.detach().cpu().numpy().shape)
#     print('inputs', np.unique(inputs.detach().cpu().numpy()), inputs.detach().cpu().numpy().shape)
#     print('targets', np.unique(targets.detach().cpu().numpy()))
#     print(pred.shape)
#     pred = pred / torch.sum(pred, -1, keepdim=True)
#     p = torch.softmax(inputs, axis=0)
#     ce_loss = xentropy_loss(targets, inputs)
    
    p = torch.sigmoid(inputs)
#     print('p-sig', np.unique(p.detach().cpu().numpy()), p.detach().cpu().numpy().shape)
#     print('sig p: ', p)
#     print(inputs.shape)
#     ce_loss = xentropy_loss(targets, inputs)
#     print('xent', np.unique(ce_loss.detach().cpu().numpy()))

    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    
#     print('ce', np.unique(ce_loss.detach().cpu().numpy()), ce_loss.detach().cpu().numpy().shape)

    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
#     print(loss)
    return loss


def gauss(images):
    cont = []
    for img in images:
#         print(img.shape)
        
        img = img.cpu().detach().numpy() 
        img = np.uint8(img)
        kernel = np.ones((3,3),np.uint8) # kernel for erosion
        g = cv2.GaussianBlur(img,(3, 3),cv2.BORDER_DEFAULT)
        cont.append(g)
    return torch.from_numpy(np.array(cont)).cuda().float()
    
def contour(images):
    '''
    Simple edge extractions:
        - Erosion
        - Then subtraction
    '''
    cont = []
#     images = images.cpu().detach().numpy() 
        
    for img in images:
#         print(img.shape)
        
        img = img.cpu().detach().numpy() 
        img = np.uint8(img)
        kernel = np.ones((3,3),np.uint8) # kernel for erosion
        r= cv2.erode(img, kernel, iterations=1) #erode

        #     #Edge extraction
        e = img - r # original_image - eroded_image
        e[e>1] = 1.0
        cont.append(e)
    return torch.from_numpy(np.array(cont)).cuda().float()
    
#############################################################################
def boundary(true, pred):
    def contour(img):
        '''
        Simple edge extractions:
            - Erosion
            - Then subtraction
        '''
        img = img.cpu().detach().numpy() 
        kernel = np.ones((3,3),np.uint8) # kernel for erosion
        r= cv2.erode(img, kernel, iterations=1) #erode

    #     #Edge extraction
        e = img - r # original_image - eroded_image
        e[e>1] = 1.0
        return torch.from_numpy(e).cuda()
    t = contour(true)
#     print(type(t), type(pred))
    p = t * pred
#     p = contour(pred)
#     return (-1.0/batch)*((t*p).mean())
#     return (-1.0)*((t-p)*(t-p)).mean()
    return ((t-p).mean())
#     return mse_loss(t, p)

#############################################################################



#############################################################################
def sp(pred):
    
    
    def contour(inst):
        k = np.ones((3,3),np.uint8)
        r= cv2.erode(np.uint8(inst), k, iterations=1)

        #Edge extraction
        e=inst-r

        return e


   ##################################################
# this cell calculates convolution for all shape priors
# and saves them all in a list
###################################################
    p_b = pred.size()[0]
    sp_path = './shapePriors/shape_priors/cpm17_final_0p44_sp/'
    sp_full_path = './shapePriors/cpm17_sp.npy'
    
    cont_arr = []
#     print(pred.size())
    for pr in pred:
#         print(pr.size(), type(pr))
        cont_arr.append(contour(pr.cpu().detach().numpy()))
    for idx, cont2 in enumerate(cont_arr):
        if idx == 0:
            cont = torch.from_numpy(cont2*1.0).unsqueeze(0) 
        else:
            cont2 = torch.from_numpy(cont2*1.0).unsqueeze(0)
            cont = torch.cat((cont, cont2), dim=0)
#     cont = [(torch.from_numpy(cont*1.0).unsqueeze(0)).unsqueeze(1) for cont in cont_arr] #converting to float tensor
    cont = cont.unsqueeze(1)
    conv_cont = []
    conv_cont_npy = []

    arr = []
    summ = torch.tensor(0.0)
    
    kernel = np.load(sp_full_path)/255.0
    kernel = torch.from_numpy(kernel).float() 
    kernel = kernel.unsqueeze(1)
    b, c, kh, kw = kernel.size()
#     kernel = kernel.repeat(p_b, 1, 1, 1)
    kernel.requires_grad_(True)
    con = F.conv2d(cont, kernel, stride= 1, padding = ((kh-1)//2,(kw-1)//2))
#     print(con.size())
    con = (con*con).mean()
    return (-1.0/(b*p_b)) * (con)

#     for idx, fil in enumerate(glob.glob(sp_path+'*.npy')):
# #         print(fil)
        
#         arr.append(np.load(fil)/255.0)
# #         print(np.unique(arr[idx]), len(np.unique(arr[idx])))
#         cont.requires_grad_(True)

#         kernel = torch.from_numpy(arr[idx]).float() 
#         kernel = (kernel.unsqueeze(0)).unsqueeze(1)
#         b, c, kh, kw = kernel.size()
#         kernel = kernel.repeat(p_b, 1, 1, 1)
#         kernel.requires_grad_(True)
#         con = F.conv2d(cont, kernel, stride= 1, padding = ((kh-1)//2,(kw-1)//2)).sum()
# #         print('con: ', con)
#         summ += (con * con)
# #         conv_cont.append(F.conv2d(cont, kernel, stride= 1, padding = ((kh-1)//2,(kw-1)//2)).sum())
# #         print(summ)
# #         conv_cont_npy.append(conv_cont[idx].detach().numpy())
        
######.cpu()#######################################################################
#     return -1.0 * summ
#     return (conv_cont * conv_cont).mean()

####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
#     print('true: ', true.size(), torch.log(pred).size())
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        if len(hv.size())<=3:
            hv = hv.unsqueeze(0)

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss


################################################################################
def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

#     print('unique: ', np.unique(img_gt))
#     img_gt = img_gt.astype(np.uint8)
#     print(np.unique(img_gt))
    
    normalized_sdf = np.zeros(out_shape)
    print(img_gt.shape, normalized_sdf.shape)
    
    for b in range(out_shape[0]): # batch size
            # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b][c] = sdf
            print(sdf.shape)
            assert np.min(sdf) == -1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
            assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
#     print('unique: ', np.unique(img_gt))
    img_gt = img_gt.astype(np.uint8)
#     print(np.unique(img_gt))
    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis - posdis
            sdf[boundary==1] = 0
            gt_sdf[b][c] = sdf

    return gt_sdf

def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: softmax results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:,1,...]
    dc = gt_sdf[:,1,...]
#     print('pc dc op_soft gt_sdf', pc.size(), dc.size(), outputs_soft.size(), gt_sdf.size())
#     multipled = torch.einsum('bxyz, bxyz->bxyz', pc.unsqueeze(-1), dc.unsqueeze(-1))
    multipled = torch.einsum('bxy, bxy->bxy', pc, dc)
    return multipled.mean()

################################################################################

class FocalLoss_Ori(torch.nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.25, 0.75], gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target.type(torch.int64)).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0, target.view(-1))
            logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
    
class FocalLoss2d(torch.nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()