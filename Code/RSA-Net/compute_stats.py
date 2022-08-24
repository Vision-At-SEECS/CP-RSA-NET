
import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)


def run_nuclei_type_stat(pred_dir, true_dir, type_uid_list=None, exhaustive=False, dataset="consep"):
    tp1_count = [0, 0, 0, 0]
    tp_count = [0, 0, 0, 0, 0, 0, 0]
    z_counter = 0
    z_files = []
    """GT must be exhaustively annotated for instance location (detection).

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image. 
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types
                     
    """
    file_list = glob.glob(pred_dir + "*.mat")
    file_list.sort()  # ensure same order [1]

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point
    
    ######################################## (start)
    type_dict = []
    empty_counter = 0
    empty_images = []
    ######################################## (end)
    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]
        print(basename)
        ######################################## (start)
        file_list = []
        file_list.append(basename)
        ######################################## (end)
        
        true_info = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]*1.0).astype("int32")
#         if dataset == "pannuke":
#             true_inst_type = np.transpose(true_inst_type)  #uncomment for pannuke
#             print(true_inst_type.shape)
#             print(true_centroid.shape)
        if len(true_info["inst_type"])==0:
            print("skipping no nuclear images ")
            empty_counter += 1
            empty_images.append(basename)
            continue
        if 0 in np.unique(true_inst_type):
            print('\n \n', basename)
            z_counter += 1
            z_files.append(basename)
            continue
#         if 0 in np.unique(true_inst_type):
#             print('\n \n', basename)
#             continue
#         print(np.unique())
        
        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[-1, -1]])
            true_inst_type = np.array([-1])
            
        for i in range(6):
            tp_count[i] += len(np.where(true_inst_type == i+1)[0])
        
        # * for converting the GT type in CoNSeP
        if dataset == "consep":
            true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 3
            true_inst_type[(true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 4
        
        tp1_count[0] += len(np.where(true_inst_type == 1)[0])
        tp1_count[1] += len(np.where(true_inst_type == 2)[0])
        tp1_count[2] += len(np.where(true_inst_type == 3)[0])
        tp1_count[3] += len(np.where(true_inst_type == 4)[0])
        
        
        pred_info = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]*1.0).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12
        )

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        
#         print(np.unique())
        
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)
        
#         print('unpaired_true: ',  len(unpaired_true_all))
#         print('unpaired_pred_all', len(unpaired_pred_all))
        
        ######################################## (start)
        file_list.append(unpaired_true)
        file_list.append(unpaired_pred)
        type_dict.append(file_list)
        ######################################## (end)
#     print(type_dict)
#     print(len(paired_all))
    paired_all = np.concatenate(paired_all, axis=0)
#     print(paired_all.shape)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)
#     print(np.unique(true_inst_type_all), true_inst_type_all.shape)
    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        print(len(true_inst_type_all), true_inst_type_all, np.unique(true_inst_type_all), np.where(true_inst_type_all ==0), len(np.where(true_inst_type_all ==0)[0]))
        type_uid_list = np.unique(true_inst_type_all).tolist()
    print(np.unique(true_inst_type_all))
    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        print("type_uid: ", type_uid, len(type_uid_list))
        print(paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w, '\n')
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(np.array(results_list))
    if dataset == "consep":
        meas = ["fd", "fa", "fm", "fi", "fe", "fs"] #consep
    elif dataset == "pannuke":
        meas = ["fd", "fa", "f_neo", "f_inf", "f_con", "f_dead", "f_epi"] #PanNuke
    else:
        print("the dataset is not registered!")
    dic = {
        "measures": meas,
        "results": results_list
    }
    print('zero_images: ', z_counter, z_files, '\n \n' ,len(meas), len(results_list))
    
#     for idx, m in enumerate(meas):
#         dic[m] = [results_list[idx]]
    print(dic, pred_dir)
    print('all categories, 4 types', tp_count, tp1_count)
    save_csv(dic, pred_dir , name="type_")
    print(empty_counter, " images were empty" , empty_images)
    return

###################################################################### For CSV (START)

def save_csv(tt_dic, save_dir, name="instance_"):
    import pandas as pd
    
    df = pd.DataFrame(tt_dic)
    df.to_csv(r'{}{}_evaluation_wrt_tissue_types.csv'.format(save_dir[:-4], name), index=False, header=True) #converting data frame to csv
#     df.to_csv(r'{}evaluation_wrt_tissue_types_noBlank.csv'.format(save_dir[:-4]), index=False, header=True) #converting data frame to csv

def save_arr_csv(array, save_dir, name="tissue_types_"):
    import pandas as pd
    
    df = pd.DataFrame(array)
    df.to_csv(r'{}{}test_image_based_measures.csv'.format(save_dir[:-4], name), index=False, header=True) #converting data frame to csv
#     df.to_csv(r'{}evaluation_wrt_tissue_types_noBlank.csv'.format(save_dir[:-4]), index=False, header=True) #converting data frame to csv

def tissue_metrics(basenames, metrics):
    
    measures = [
            'dice',
            'fast_aji',
            'detection_quality',
            'segmentation_quality',
            'panoptic_quality',
            'fast_aji_plus',
        ] #measures on which the network was evaluated
    
    
    
    tt_dic = {
        'measures': measures,
    }
    
    for tt in np.unique(basenames):
        print(np.unique(basenames))
        indexes =  [index for index, element in enumerate(basenames) if element == tt] #indexes of metrics of a particular tissue type

#         print(tt, indexes)
        tt_metrics = [] #new metrics wrt tt
        print(len(metrics[0]), len(indexes))
        print(indexes)
            
        for metlen in metrics:
            mtr = [metlen[i] for i in indexes]
            tt_metrics.append(mtr) #keeping metrics of only a particular tissue type
        
        tt_metrics = np.array(tt_metrics)
        print(len(indexes), len(tt_metrics[0]))
        
        metrics_avg = np.mean(tt_metrics, axis=-1)
        
        np.set_printoptions(formatter={"float": "{: 0.5f}".format})
        print(' tissue metrics: ', metrics_avg)
        metrics_avg = list(metrics_avg)
        
        tt_dic[tt] = metrics_avg
    print('bbb: ', basenames)
#     print('tissue: ', len(tt_metrics))
    
    return tt_dic
 
###################################################################### For CSV (END)
def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False, ext=".mat"):
    
    # print stats of each image

    print(pred_dir)

    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()  # ensure same order

    metrics = [[], [], [], [], [], []]
    z_counter = 0
    z_files = []
    ################################################################### (START)
    ### names
    image_wise_measures = [[
        'image_name',
        'dice',
        'fast_aji',
        'detection_quality',
        'segmentation_quality',
        'panoptic_quality',
        'fast_aji_plus',
    ]]

    basenames = []
    for idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

    ################################################################### (END)
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]
        print(basename)
        
        true = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        true = (true["inst_map"]).astype("int32")
        print(np.unique(true))
        if len(np.unique(true)) <= 1:
            z_counter += 1
            z_files.append(basename)
            print(basename, ' ', z_counter)
            continue
        
        
        pred = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        pred = (pred["inst_map"]).astype("int32")
        print(np.unique(pred))

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(get_fast_aji_plus(true, pred))

        if print_img_stats:
            print(basename, end="\t")
            ######################################################## (START)
            image_scores = [basename]
            ######################################################## (END)
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
                image_scores.append(scores[-1])
            print('empty_images: ', z_counter)    
######################################################## (START)
            image_wise_measures.append(image_scores)
        basenames.append(basename.split('_')[0])
    
    tt_dic = tissue_metrics(basenames, metrics)
    print(tt_dic)


######################################################## (END)
    ####
    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(metrics_avg)
    metrics_avg = list(metrics_avg)
    
    
######################################################## (START)    
    tt_dic['overall_score'] = metrics_avg
    save_csv(tt_dic, pred_dir)
    save_arr_csv(image_wise_measures, pred_dir)
######################################################## (END)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="mode to run the measurement,"
        "`type` for nuclei instance type classification or"
        "`instance` for nuclei instance segmentation",
        nargs="?",
        default="instance",
        const="instance",
    )
    parser.add_argument(
        "--pred_dir", help="point to output dir", nargs="?", default="", const=""
    )
    parser.add_argument(
        "--true_dir", help="point to ground truth dir", nargs="?", default="", const=""
    )
    parser.add_argument(
        "--dataset", help="dataset for type setting", nargs="?", default="consep", const=""
    )
    args = parser.parse_args()

    if args.mode == "instance":
        run_nuclei_inst_stat(args.pred_dir, args.true_dir, print_img_stats=True)
    if args.mode == "type":
        run_nuclei_type_stat(args.pred_dir, args.true_dir, dataset= args.dataset)
