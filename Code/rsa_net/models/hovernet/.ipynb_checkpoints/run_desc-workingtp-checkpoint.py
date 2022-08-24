import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from misc.utils import center_pad_to_shape, cropping_center
from .utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss

from collections import OrderedDict

count = 0
####
def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to("cuda").type(torch.int64)
    true_hv = true_hv.to("cuda").type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    pred_dict = model(imgs)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    if model.module.nr_types is not None:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    ####
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot[..., 1])
            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    ####

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, true_np.shape[0], (2,))

    imgs = (imgs[sample_indices]).byte()  # to uint8
    imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
    pred_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    }

    true_dict["np"] = true_np
    true_dict = {
        k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    }

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs,
        "np": (true_dict["np"], pred_dict["np"]),
        "hv": (true_dict["hv"], pred_dict["hv"]),
    }
    return result_dict


####
def valid_step(batch_data, run_info):
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = torch.squeeze(true_np).to("cuda").type(torch.int64)
    true_hv = torch.squeeze(true_hv).to("cuda").type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    if model.module.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_dict["tp"] = true_tp

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
        if model.module.nr_types is not None:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].cpu().numpy(),
            "true_hv": true_dict["hv"].cpu().numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
        }
    }
    if model.module.nr_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].cpu().numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()
    return result_dict


####
def infer_step(batch_data, model):
    RTS = True
    GC = False
#     batch_data, sample_info_list = batch_data
#     np.save('np.npy', sample_info_list)
    ####\
    
    mask = batch_data[..., -1]
    batch_data = batch_data[..., :3]
    patch_imgs = batch_data
    
    print('batch data size ', batch_data.size())
#     true_np = batch_data["np_map"]
    
    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()
    
    ####
    model.eval()  # infer mode\
    if GC:
        gradcams = segm_gc(model, patch_imgs_gpu, mask)
        
#     print(model)
    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        
        if RTS:
            pred_dict = RTS_testTime(model, patch_imgs_gpu, batch_data)
        else: 
            pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
#         if "tp" in pred_dict:
#             type_map = F.softmax(pred_dict["tp"], dim=-1)
#             type_map = torch.argmax(type_map, dim=-1, keepdim=True)
#             type_map = type_map.type(torch.float32)
#             pred_dict["tp"] = type_map
        print("tp in calc: ", pred_dict["tp"].size())
        pred_output = torch.cat(list(pred_dict.values()), -1)
        print(pred_output.size())
        print(np.unique(pred_dict["tp"].cpu().numpy()) )
#         np.save('pred_output.npy', pred_output.cpu().numpy())
#         np.save('np_RTS.npy', pred_dict["np"].cpu().numpy())
    # * Its up to user to define the protocol to process the raw output per step!
    if GC:
        return pred_output.cpu().numpy(), gradcams
    else:
        return pred_output.cpu().numpy()

####
def viz_step_output(raw_data, nr_types=None):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    true_np, pred_np = raw_data["np"]
    true_hv, pred_hv = raw_data["hv"]
    if nr_types is not None:
        true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)

        true_viz_list = [img]
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
        true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
        if nr_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, nr_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img]
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
        pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
        if nr_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, nr_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


####
from itertools import chain


def proc_valid_step_output(raw_data, nr_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data["prob_np"]
    true_np = raw_data["true_np"]
    for idx in range(len(raw_data["true_np"])):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_value("np_acc", acc_np, "scalar")
    track_value("np_dice", dice_np, "scalar")

    # * TP statistic
    if nr_types is not None:
        pred_tp = raw_data["pred_tp"]
        true_tp = raw_data["true_tp"]
        for type_id in range(0, nr_types):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["true_np"])):
                patch_pred_tp = pred_tp[idx]
                patch_true_tp = true_tp[idx]
                inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
                over_inter += inter
                over_total += total
            dice_tp = 2 * over_inter / (over_total + 1.0e-8)
            track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # * HV regression statistic
    pred_hv = raw_data["pred_hv"]
    true_hv = raw_data["true_hv"]

    over_squared_error = 0
    for idx in range(len(raw_data["true_np"])):
        patch_pred_hv = pred_hv[idx]
        patch_true_hv = true_hv[idx]
        squared_error = patch_pred_hv - patch_true_hv
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_value("hv_mse", mse, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    true_np = np.array([true_np[idx] for idx in selected_idx])
    true_hv = np.array([true_hv[idx] for idx in selected_idx])
    prob_np = np.array([prob_np[idx] for idx in selected_idx])
    pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs, "np": (true_np, prob_np), "hv": (true_hv, pred_hv)}

    if nr_types is not None:
        true_tp = np.array([true_tp[idx] for idx in selected_idx])
        pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
        viz_raw_data["tp"] = (true_tp, pred_tp)
    viz_fig = viz_step_output(viz_raw_data, nr_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict



from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM 
import cv2

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
#         self.mask = torch.from_numpy(mask)
        self.mask = mask
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        print('in target init')
    def __call__(self, model_output):
        print('in target call')
        return (model_output[self.category, :, : ] * self.mask).sum()

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        print('in wrappers init')
    def forward(self, x):
        pred_dict = self.model(x)
#         pred_dict = OrderedDict(
#             [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
#         )
#         pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
#         print('prediction size: ', pred_dict["np"][:,1,...].shape)
        return pred_dict["np"][:,1,...].unsqueeze(0)
#         return pred_dict[:,1,...]


def segm_gc(model, input_tensor, mask):
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
#     print(model)
    model =  SegmentationModelOutputWrapper(model)
    output = model(input_tensor)
    
    
#     target_layers = [model.model.module.d3]
    target_layers = [model.model.module.d3.units[-1]]
    print('output size: ', output.size())
    print('mask', mask.size(), mask[:,(270//2)-40:(270//2)+40, (270//2)-40:(270//2)+40].shape)
    targets = [SemanticSegmentationTarget(0, mask[:,(270//2)-40:(270//2)+40, (270//2)-40:(270//2)+40].unsqueeze(0))]
#     print(targets[0].shape)
    print('input_tensor: ', input_tensor.cpu().numpy().shape)
    
    with GradCAM(model=model,
                 target_layers=[model.model.module.d3.units[-1]],
                 use_cuda=torch.cuda.is_available()) as cam:
#                  use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor=input_tensor # ) # ,
                            , targets=targets)[0, :]
        print('grayscale_cam', grayscale_cam.shape, np.unique(grayscale_cam))
        np.save("/home/nimra_amin/WORKSPACE/JupyterLab/nucleiDetectClassify/hover_net/grayscale_img.npy", grayscale_cam)
#         grayscale_cam = np.expand_dims(grayscale_cam, axis=0)
#         grayscale_cam = np.expand_dims(grayscale_cam, axis=0)
#         image = np.float32(input_tensor.reshape((270, 270, 3)).cpu().numpy()) / 255
        image = crop_to_shape(input_tensor, torch.tensor(np.ones((1,3,80,80))) )
        image = np.transpose(image.squeeze(0).cpu().numpy(), (1, 2, 0))/255.0
#         image = image[(270//2)-40:(270//2)+40, (270//2)-40:(270//2)+40,:]
#         grayscale_cam = cv2.resize(grayscale_cam, (80,80), interpolation = cv2.INTER_AREA)
        grayscale_cam = grayscale_cam[((270//2)-40):((270//2)+40), ((270//2)-40):((270//2)+40)]
        
        cam_image = show_cam_on_image( image, grayscale_cam, use_rgb=True)
        
#         cv2.imwrite("/home/nimra_amin/WORKSPACE/JupyterLab/nucleiDetectClassify/hover_net/dataloader/cam_img.png", cam_image)
        np.save("/home/nimra_amin/WORKSPACE/JupyterLab/nucleiDetectClassify/hover_net/cam_img.npy", cam_image)
    return (cam_image, grayscale_cam)
#     Image.fromarray(cam_image)


import ttach as tta
import collections 


from imgaug import augmenters as iaa

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)
from scipy import stats

def RTS_testTime(model, image, patch_images=None):
    rng = 7
    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[90]),
#         tta.Scale(scales=[1, 2, 4]),
#         tta.Multiply(factors=[0.9, 1.1]),        
    ]
    )
    # Example how to process ONE batch on images with TTA
# Here `image`/`mask` are 4D tensors (B, C, H, W), `label` is 2D tensor (B, N)
    masks_np = []
    masks_hv = []
    deaug_mask = {
        "np": [],
        "hv": []
    }
    print(len(transforms))
    for idx, transformer in enumerate(transforms): # custom transforms or e.g. tta.aliases.d4_transform() 
        print(idx, transforms, transformer)
        # augment image
        augmented_image = transformer.augment_image(image)
        model_output = model(augmented_image)
#         model_output = OrderedDict(
#             [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in model_output.items()]
#         )
#         model_output["np"] = F.softmax(model_output["np"], dim=-1)[..., 1:]
#         model_output = OrderedDict(
#             [[k,  transformer.deaugment_mask(v)] for k, v in model_output.items()]
#         )
#         model_output = transformer.deaugment_mask(model_output)
#         print('deaugmenting image right after the ', model_output.size())
#         TODO: Reduce the below bogus lengthy logic, that was written in a haste

        if idx==0: #this is the original image with no transformations
                 
            model_output_np =  model_output['np'].unsqueeze(0)  
            model_output_hv =  model_output['hv'].unsqueeze(0)
            
            
            if "tp" in model_output:
                model_output['tp'] = model_output['tp'].permute(0, 2, 3, 1).contiguous()
                type_map = F.softmax( model_output['tp'], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                model_output_tp = type_map.type(torch.float32)
                model_output_tp = model_output_tp.permute(0, 3, 1, 2).contiguous() 
                model_output_tp =  model_output_tp.unsqueeze(0)
                
#                 print('after deaugmentation tp ', type(model_output['tp']), model_output_tp.size())
                
                
#             model_output_np = transformer.deaugment_mask(model_output_np)
#             model_output_hv = transformer.deaugment_mask(model_output_hv)
#             print('deaugmented mask size: ',  model_output_np.size() , model_output_hv.size())
            
            model_output_np1 = model_output_np[:,:,0,...]  
            model_output_hv1 =  model_output_hv[:,:,0,...]
            
            model_output_np2 = model_output_np[:,:,1,...]
            model_output_hv2 =  model_output_hv[:,:,1,...]
            
            
#             deaug_mask_np1 = transformer.deaugment_mask(model_output_np1)
#             deaug_mask_np2 = transformer.deaugment_mask(model_output_np2)

#             deaug_mask_hv1 = transformer.deaugment_mask(model_output_hv1)
#             deaug_mask_hv2 = transformer.deaugment_mask(model_output_hv2)
             
            
            
        else: #for concatenation a lame logic that should be changed in the future
            
#             print('before deaugmentation mask np,hv: ',  model_output['np'].size() , model_output['hv'].size())
            model_output_np = transformer.deaugment_mask(model_output['np'])
            model_output_hv = transformer.deaugment_mask(model_output['hv'])
#             print('deaugmented mask size np,hv: ',  model_output_np.size() , model_output_hv.size())
            
            model_output_np  = model_output_np.unsqueeze(0)  
            model_output_hv =  model_output_hv.unsqueeze(0)
            if "tp" in model_output:
#                 print('to before deaug tp ', type(model_output['tp']),model_output['tp'].size())  
                model_output_tp2 = transformer.deaugment_mask(model_output['tp'])
                model_output_tp2 = model_output_tp2.permute(0, 2, 3, 1).contiguous()        
                print('tp2', model_output_tp2.size())
                type_map = F.softmax( model_output_tp2, dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                model_output_tp2 = type_map.type(torch.float32)
                model_output_tp2 = model_output_tp2.permute(0, 3, 1, 2).contiguous() 
                model_output_tp2 = model_output_tp2.unsqueeze(0)[...]
#                 print('after deaug tp: ', model_output_tp2.size(), model_output_tp.size())
                model_output_tp = torch.cat((model_output_tp, model_output_tp2), axis=0)
                print('cat:', model_output_tp.size())
        
            model_output_np3 = model_output_np[:,:,0,...]  
            model_output_hv3 =  model_output_hv[:,:,0,...]
            
            model_output_np4 = model_output_np[:,:,1,...]
            model_output_hv4 =  model_output_hv[:,:,1,...]
            
#             model_output_np3 = transformer.deaugment_mask(model_output_np3)
#             model_output_np4 = transformer.deaugment_mask(model_output_np4)

#             model_output_hv3 = transformer.deaugment_mask(model_output_hv3)
#             model_output_hv4 = transformer.deaugment_mask(model_output_hv4)
            
            model_output_np1 = torch.cat((model_output_np1, model_output_np3), axis = 0)
            model_output_np2 = torch.cat((model_output_np2, model_output_np4), axis = 0)
            
            model_output_hv1 = torch.cat((model_output_hv1, model_output_hv3), axis = 0)        
            model_output_hv2 = torch.cat((model_output_hv2, model_output_hv4), axis = 0)  
                
#         deaug_mask_np1 = transformer.deaugment_mask(model_output_np1)
#         deaug_mask_np2 = transformer.deaugment_mask(model_output_np2)
        
#         deaug_mask_hv1 = transformer.deaugment_mask(model_output_hv1)
#         deaug_mask_hv2 = transformer.deaugment_mask(model_output_hv2)
    
    image_augs = [
                    iaa.GaussianBlur(sigma=(0.0, 3.0)),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
                    
                    ]
    for img_aug in image_augs:
        print('pi:', patch_images.size())
        augmented_image = img_aug(images=image.cpu().numpy())
        print(augmented_image.shape)
        patch_imgs_gpu = torch.tensor(augmented_image).to("cuda").type(torch.float32)  # to NCHW
#         patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()
        model_output = model(patch_imgs_gpu)
        # pass to model
          # save results
            
        model_output_np  = model_output['np'].unsqueeze(0)  
        model_output_hv =  model_output['hv'].unsqueeze(0)
        if "tp" in model_output:
            
            model_output['tp'] = model_output['tp'].permute(0, 2, 3, 1).contiguous()
            type_map = F.softmax( model_output['tp'], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            model_output_tp2 = type_map.type(torch.float32)
            model_output_tp2 = model_output_tp2.permute(0, 3, 1, 2).contiguous() 
            model_output_tp2 = model_output_tp2.unsqueeze(0)[...]
            model_output_tp = torch.cat((model_output_tp, model_output_tp2), axis=0)
                
        model_output_np3 = model_output_np[:,:,0,...]  
        model_output_hv3 =  model_output_hv[:,:,0,...]

#         model_output_np4 = model_output_np[:,:,1,...]
        model_output_hv4 =  model_output_hv[:,:,1,...]
        print(model_output_np1.size(), model_output_np3.size())
        model_output_np1 = torch.cat((model_output_np1, model_output_np3), axis = 0)
        model_output_np2 = torch.cat((model_output_np2, model_output_np4), axis = 0)

        model_output_hv1 = torch.cat((model_output_hv1, model_output_hv3), axis = 0)        
        model_output_hv2 = torch.cat((model_output_hv2, model_output_hv4), axis = 0)    
        
           

    mean_mask = {
        "tp": torch.tensor(stats.mode(model_output_tp.cpu().numpy())[0]).squeeze(0).to("cuda"),
        "np": torch.cat((torch.mean(model_output_np1, axis=0).unsqueeze(1), torch.mean(model_output_np2, axis=0).unsqueeze(1)), axis=1),
        "hv": torch.cat((torch.tensor(np.maximum(model_output_hv1.cpu().numpy()[0,...], model_output_hv1.cpu().numpy()[0,...])).unsqueeze(1),
                         torch.tensor(np.maximum(model_output_hv2.cpu().numpy()[0,...], model_output_hv2.cpu().numpy()[0,...])).unsqueeze(1)), axis=1).to("cuda"),
        
    }
#     if "tp" in model_output:
#         mean_mask["tp"]  = torch.tensor(stats.mode(model_output_tp.cpu().numpy())[0]).squeeze(0).to("cuda")
#         print(np.where(stats.mode(model_output_tp.cpu().numpy())[1]==1), np.unique(stats.mode(model_output_tp.cpu().numpy())[1]) )
#         print(np.unique(mean_mask["tp"].cpu().numpy()))
#         print('tp shape: ', mean_mask["tp"].size())
    
#     print(mean_mask['hv'].size())
#     print(np.where(stats.mode(model_output_tp.cpu().numpy())[1]==1))
    _, b, ch, row, col = np.where(stats.mode(model_output_tp.cpu().numpy())[1]==1)
    mean_mask["tp"][b,ch,row,col] = model_output_tp[0,b,ch,row,col]
#     np.save('./mean_mask_np.npy', mean_mask["np"].cpu().numpy())
    
    return mean_mask
    