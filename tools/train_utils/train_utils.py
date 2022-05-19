import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.datasets import build_active_dataloader
from pcdet.utils import common_utils, commu_utils

import pickle
def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None, lr_scheduler=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    if lr_scheduler is not None and hasattr(lr_scheduler, 'state_dict'):
        lr_scheduler_state = lr_scheduler.state_dict()
    else:
        lr_scheduler_state  = None
    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version, 'lr_scheduler': lr_scheduler_state}


def save_checkpoint(state, filename='checkpoint'):
    # if False and 'optimizer_state' in state:
    #     optimizer_state = state['optimizer_state']
    #     state.pop('optimizer_state', None)
    #     optimizer_filename = '{}_optim.pth'.format(filename)
    #     torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def resume_datset(labelled_loader, unlabelled_loader, selected_frames_list, dist_train, logger, cfg):

    labelled_set = labelled_loader.dataset
    unlabelled_set = unlabelled_loader.dataset


    if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
        pairs = list(zip(unlabelled_set.sample_id_list, unlabelled_set.kitti_infos))
    else:
        pairs = list(zip(unlabelled_set.frame_ids, unlabelled_set.infos))

    # get selected frame id list and infos
    if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
        selected_id_list, selected_infos = list(labelled_set.sample_id_list), \
                                            list(labelled_set.kitti_infos)
        unselected_id_list, unselected_infos = list(unlabelled_set.sample_id_list), \
                                                list(unlabelled_set.kitti_infos)
    else: # Waymo
        selected_id_list, selected_infos = list(labelled_set.frame_ids), \
                                                    list(labelled_set.infos)
        unselected_id_list, unselected_infos = list(unlabelled_set.frame_ids), \
                                                        list(unlabelled_set.infos)

    selected_frames_files = []
    for file_dir in selected_frames_list:
        pkl_file = pickle.load(open(file_dir, 'rb'))
        frame_list = [str(i) for i in pkl_file['frame_id']]
        selected_frames_files += frame_list
        
        print('successfully load the selected frames from {}'.format(file_dir.split('/')[-1]))

    for i in range(len(pairs)):
        if pairs[i][0] in selected_frames_files:
            selected_id_list.append(pairs[i][0])
            selected_infos.append(pairs[i][1])
            unselected_id_list.remove(pairs[i][0])
            unselected_infos.remove(pairs[i][1])

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
    
