import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"
        model_mode = "original" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = None #5 #5 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = False #True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [80, 80] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `original` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "consep"#"cpm17" # extracts dataset info from dataset.py
        self.log_dir = "logs/consep_270_80/consep_bn_class" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/consep_bn_class/011/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/skipCBAM-ECAk7_consep_bn_class/01_3bs/" # where checkpoints will be saved

        self.log_dir = "logs/consep_270_80/skipCBAM-ECAk7_consep_bn_class/amp_dataset/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/skipCBAM-ECAk7_consep_bn_class_3b/" # where checkpoints will be saved
        
        self.log_dir = "logs/consep_270_80/consep_bn/01/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/CBAMECAk7_consep_bn/new/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/skip2CBAMECAk7_consep_bn/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/skipCBAMECA-kist-sharedSpat_consep_bn/02/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/CoNSeP_bn/02/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/skipCBAMECA-kist-sharedSpat_consep_bn/03/" # where checkpoints will be saved
        self.log_dir = "logs/consep_270_80/CBAMECAk5_consep_bn/03/" # where checkpoints will be saved
        
#         ####################33 CPM17
        self.log_dir = "logs/cpm17/CBAMECAk5_sharedSpat_cpm_SELU/" # where checkpoints will be saved
        
        self.log_dir = "logs/cpm17/CBAMECAk5_cpm_SELU/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/sepCBAMECAk5_cpm_SELU/00/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/sepCBAMECAk5_sharedSpat_cpm17_ELU-BN/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/sepCBAMECAk5_sharedSpat_cpm17_RELU-BN/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/sharedCBAMECAk5_sharedSpat_cpm17_RELU-BN/" # where checkpoints will be saved
        
        self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN/01/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/sharedCBAMECAk5_cpm17_RELU-BN/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/cpm17_bn/try2/001" # where checkpoints will be saved

        self.log_dir = "logs/CoNSeP_270_80/CoNSeP_bn/" # where checkpoints will be saved

#         self.log_dir = "logs/cpm17/sharedCBAMECAk5_cpm17_RELU-BN_chBn/01/" # where checkpoints will be saved
#         self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_chBnSpBn/01/" # where checkpoints will be saved
#         self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_chCatBnSpBn/" # where checkpoints will be saved
#         self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_chBnSpBnCor/01/" # where checkpoints will be saved
#         self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_chBnRelu_SpBnRelu/02/" # where checkpoints will be saved
#         self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_ch_SpBnRelu/02/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_con_map-cat-npMa-1/" # where checkpoints will be saved
#         self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_con_map/" # where checkpoints will be saved
#         self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_con_mapInstance/01/" # where checkpoints will be saved
        self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_con_map-cat-npMa-bcep5-dice1/" # 
        self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_con_map-nphv-cat-npMa-bcep5-dice1/01/" # 
        
        self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_con_map-catnp-Dicep5-Bound1/" # 

        self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_RELU-BN_con_map-catnp-Dicep5-Bound1/" #
        
        self.log_dir = "logs/cpm17/sharedCBAMECAk7_cpm17_preactSP-BN-RELu/" # where checkpoints will be saved
        
        self.log_dir = "logs/CoNSeP_270_80/rec_CBAMECAk7_consep_preactrelubn/00-01/02/" # where 

        self.log_dir = "logs/CoNSeP_270_80/rec_CBAMECAk7_consep-tp_preactrelubn_fullData/msge_2/00-01/02/" # where 
        
        self.log_dir = "logs/CoNSeP_270_80/rec_CBAMECAk7_consep-tp_preactrelubn_fullData/msge_2/msge2_01_amp/" # where 

        
        self.log_dir = "logs/CoNSeP_270_80/rec_CBAMECAk5_consep-tp_preactrelubn_fullData/00_msge2_01/noAtt_01_24-50/" # where 
        
        self.log_dir = "logs/cpm17/train_on_test_sharedCBAMECAk7_cpm17_preactSP-BN-RELu/" # where 
        self.log_dir = "logs/cpm17/train_on_test_sharedCBAMECAk5_cpm17_preactSP-BN-RELu/mse2-msge2/05/" # where 
        self.log_dir = "test/" # where 
        
#         self.log_dir = "logs/CoNSeP_270_80/rec_CBAMECAk7_consep_preactrelubn_fullData/msge2_00_01/01/" # where 

        
        # paths to training and validation patches
        self.train_dir_list = [
#             "../DATASETS/training_data/consep_class/consep/train/540x540_80x80/"
#             "../DATASETS/training_data/cpm/cpm17/test/540x540_80x80/"
            "../DATASETS/training_data/consep_class/consep/test/540x540_80x80"
        ]
        self.valid_dir_list = [
#             "../DATASETS/training_data/consep/consep/test/540x540_80x80/"
#             "../DATASETS/training_data/cpm/cpm17/train/540x540_80x80/"
             "../DATASETS/training_data/consep_class/consep/test/540x540_80x80"
        ]
        
        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
