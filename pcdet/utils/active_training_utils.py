from itertools import accumulate
import torch
import os
import glob
import tqdm
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
import pickle as pkl
import re
from pcdet.datasets import build_active_dataloader
from .. import query_strategies
ACTIVE_LABELS = {}
NEW_ACTIVE_LABELS = {}


def check_already_exist_active_label(active_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        active_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.ACTIVE_TRAIN.get('INITIAL_SELECTION', None):
        if os.path.exists(cfg.ACTIVE_TRAIN.INITIAL_SELECTION):
            init_active_label = pkl.load(open(cfg.ACTIVE_TRAIN.INITIAL_SELECTION, 'rb'))
            ACTIVE_LABELS.update(init_active_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(active_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(ACTIVE_LABELS, f)

            return cfg.ACTIVE_TRAIN.INITIAL_SELECTION

    active_label_list = glob.glob(os.path.join(active_label_dir, 'active_label_e*.pkl'))
    if len(active_label_list) == 0:
        return

    active_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in active_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            ACTIVE_LABELS.update(latest_ps_label)
            return cur_pkl

    return None

def save_active_label_epoch(model, val_loader, rank, leave_pbar, active_label_dir, cur_epoch):
    """
    Generate active with given model.

    Args:
        model: model to predict result for active label
        val_loader: data_loader to predict labels
        rank: process rank
        leave_pbar: tqdm bar controller
        active_label_dir: dir to save active label
        cur_epoch
    """
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_active_e%d' % cur_epoch, dynamic_ncols=True)

    # pos_ps_meter = common_utils.AverageMeter()
    # ign_ps_meter = common_utils.AverageMeter()

    model.eval()

    for cur_it in range(total_it_each_epoch):
        try:
            unlabelled_batch = next(val_dataloader_iter)
        except StopIteration:
            unlabelled_dataloader_iter = iter(val_loader)
            unlabelled_batch = next(unlabelled_dataloader_iter)

        # generate gt_boxes for unlabelled_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(unlabelled_batch)
            pred_dicts, recall_dicts = model(unlabelled_batch)

        # select and save active labels
        # random

        save_active_label_batch(
            unlabelled_batch, pred_dicts=pred_dicts
        )

        # log to console and tensorboard
        # pos_ps_meter.update(pos_ps_batch)
        # ign_ps_meter.update(ign_ps_batch)
        # disp_dict = {'pos_ps_box': "{:.3f}({:.3f})".format(pos_ps_meter.val, pos_ps_meter.avg),
        #              'ign_ps_box': "{:.3f}({:.3f})".format(ign_ps_meter.val, ign_ps_meter.avg)}

        if rank == 0:
            pbar.update()
            # pbar.set_postfix(disp_dict)
            pbar.refresh()

    if rank == 0:
        pbar.close()

    gather_and_dump_pseudo_label_result(rank, active_label_dir, cur_epoch)


def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()

    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_ACTIVE_LABELS)

        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)

        NEW_ACTIVE_LABELS.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "active_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_ACTIVE_LABELS, f)

    commu_utils.synchronize()
    ACTIVE_LABELS.clear()
    ACTIVE_LABELS.update(NEW_ACTIVE_LABELS)
    NEW_ACTIVE_LABELS.clear()


def save_active_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    batch_size = len(pred_dicts)

    for b_idx in range(batch_size):
        # pred_cls_scores = pred_iou_scores = None
        # if 'pred_boxes' in pred_dicts[b_idx]:
        #     # Exist predicted boxes passing self-training score threshold
        #     pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
        #     pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
        #     pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
        #     if 'pred_cls_scores' in pred_dicts[b_idx]:
        #         pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
        #     if 'pred_iou_scores' in pred_dicts[b_idx]:
        #         pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()
        #
        #     # remove boxes under negative threshold
        #     if cfg.SELF_TRAIN.get('NEG_THRESH', None):
        #         labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
        #         remain_mask = pred_scores >= labels_remove_scores
        #         pred_labels = pred_labels[remain_mask]
        #         pred_scores = pred_scores[remain_mask]
        #         pred_boxes = pred_boxes[remain_mask]
        #         if 'pred_cls_scores' in pred_dicts[b_idx]:
        #             pred_cls_scores = pred_cls_scores[remain_mask]
        #         if 'pred_iou_scores' in pred_dicts[b_idx]:
        #             pred_iou_scores = pred_iou_scores[remain_mask]
        #
        #     labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
        #     ignore_mask = pred_scores < labels_ignore_scores
        #     pred_labels[ignore_mask] = -1
        #
        #     gt_box = np.concatenate((pred_boxes,
        #                              pred_labels.reshape(-1, 1),
        #                              pred_scores.reshape(-1, 1)), axis=1)
        #
        # else:
        #     # no predicted boxes passes self-training score threshold
        #     gt_box = np.zeros((0, 9), dtype=np.float32)
        #
        # gt_infos = {
        #     'gt_boxes': gt_box,
        #     'cls_scores': pred_cls_scores,
        #     'iou_scores': pred_iou_scores,
        #     'memory_counter': np.zeros(gt_box.shape[0])
        # }

        # record pseudo label to pseudo label dict
        # if need_update:
        #     ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
        #     gt_infos = ensemble_func(ACTIVE_LABELS[input_dict['frame_id'][b_idx]],
        #                              gt_infos, cfg.SELF_TRAIN.MEMORY_ENSEMBLE)

        # if gt_infos['gt_boxes'].shape[0] > 0:
        #     ign_ps_meter.update((gt_infos['gt_boxes'][:, 7] < 0).sum())
        # else:
        #     ign_ps_meter.update(0)
        # pos_ps_meter.update(gt_infos['gt_boxes'].shape[0] - ign_ps_meter.val)

        NEW_ACTIVE_LABELS[input_dict['frame_id'][b_idx]] = input_dict['gt_boxes']

    return


def load_ps_label(frame_id):
    """
    :param frame_id: file name of pseudo label
    :return gt_box: loaded gt boxes (N, 9) [x, y, z, w, l, h, ry, label, scores]
    """
    if frame_id in ACTIVE_LABELS:
        gt_box = ACTIVE_LABELS[frame_id]['gt_boxes']
    else:
        raise ValueError('Cannot find pseudo label for frame: %s' % frame_id)

    return gt_box


def select_active_labels(model,
                         labelled_loader,
                         unlabelled_loader,
                         rank,
                         logger,
                         method,
                         leave_pbar=True,
                         cur_epoch=None,
                         dist_train=False,
                         active_label_dir=None,
                         accumulated_iter=None):

    strategy = query_strategies.build_strategy(method=method, model=model, \
            labelled_loader=labelled_loader, \
            unlabelled_loader=unlabelled_loader, \
            rank=rank, \
            active_label_dir=active_label_dir, \
            cfg=cfg)
    if os.path.isfile(os.path.join(active_label_dir, 'selected_frames_epoch_{}.pkl'.format(cur_epoch))):
            # if found, then resume

            # TODO: needed to be revised
            print('found {} epoch saved selections...start resuming...'.format(cur_epoch))
            with open(os.path.join(active_label_dir, 'selected_frames_epoch_{}.pkl'.format(cur_epoch)), 'rb') as f:
                unpickler = pkl.Unpickler(f)
                # if file is not empty scores will be equal
                # to the value unpickled
                selected_frames = unpickler.load(f)
                
    else:
      
        selected_frames = strategy.query(leave_pbar, cur_epoch)
        strategy.save_active_labels(selected_frames=selected_frames, cur_epoch=cur_epoch)
        strategy.update_dashboard(cur_epoch=cur_epoch, accumulated_iter=accumulated_iter)

    
    # get selected frame id list and infos
    if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
        selected_id_list, selected_infos = list(strategy.labelled_set.sample_id_list), \
                                            list(strategy.labelled_set.kitti_infos)
        unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), \
                                                list(strategy.unlabelled_set.kitti_infos)
    else: # Waymo
        selected_id_list, selected_infos = list(strategy.labelled_set.frame_ids), \
                                                    list(strategy.labelled_set.infos)
        unselected_id_list, unselected_infos = list(strategy.unlabelled_set.frame_ids), \
                                                        list(strategy.unlabelled_set.infos)

    for i in range(len(strategy.pairs)):
        if strategy.pairs[i][0] in selected_frames:
            selected_id_list.append(strategy.pairs[i][0])
            selected_infos.append(strategy.pairs[i][1])
            unselected_id_list.remove(strategy.pairs[i][0])
            unselected_infos.remove(strategy.pairs[i][1])

    selected_id_list, selected_infos, \
    unselected_id_list, unselected_infos = \
        tuple(selected_id_list), tuple(selected_infos), \
        tuple(unselected_id_list), tuple(unselected_infos)

    active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]


    # Create new dataloaders
    batch_size = unlabelled_loader.batch_size
    print("Batch_size of a single loader: %d" % (batch_size))
    workers = unlabelled_loader.num_workers

    # delete old sets and loaders, save space for new sets and loaders
    del labelled_loader.dataset, unlabelled_loader.dataset, \
        labelled_loader, unlabelled_loader

    labelled_set, unlabelled_set, \
    labelled_loader, unlabelled_loader, \
    sampler_labelled, sampler_unlabelled = build_active_dataloader(
        cfg.DATA_CONFIG,
        cfg.CLASS_NAMES,
        batch_size,
        dist_train,
        workers=workers,
        logger=logger,
        training=True,
        active_training=active_training
    )

    return labelled_loader, unlabelled_loader
