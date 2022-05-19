import torch
import os
import glob
import tqdm
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
from pcdet.utils import self_training_utils
from pcdet.utils import active_training_utils
from pcdet.config import cfg
from .train_utils import save_checkpoint, checkpoint_state, resume_datset
from train_utils.optimization import build_scheduler, build_optimizer
from train_utils.train_utils import model_state_to_cpu
import wandb
from pcdet.models import build_network

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, history_accumulated_iter=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    model.train()
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        # if tb_log is not None:
        #     tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        #     wandb.log({'meta_data/learning_rate': cur_lr}, step=accumulated_iter)

        
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        lr_scheduler.step(accumulated_iter)
        if history_accumulated_iter is not None:
            history_accumulated_iter += 1
            accumulated_iter += 1
            log_accumulated_iter = history_accumulated_iter
        else:
            accumulated_iter += 1
            log_accumulated_iter = accumulated_iter
       
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=log_accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, log_accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr,log_accumulated_iter)
                wandb.log({'train/loss': loss, 'meta_data/learning_rate': cur_lr}, step=log_accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, log_accumulated_iter)
                    wandb.log({'train/' + key: val}, step=log_accumulated_iter)
    if rank == 0:
        pbar.close()
    if history_accumulated_iter is not None:
        return accumulated_iter, history_accumulated_iter
    else:
        return accumulated_iter

def train_model_active(model, optimizer, labelled_loader, unlabelled_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, active_label_dir, backbone_dir,
                   labelled_sampler=None, unlabelled_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                   max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, dist_train=False):

    # 1. Pre-train the labelled_loader for a number of epochs (cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS)
    accumulated_iter = start_iter
    active_pre_train_epochs = cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS
    logger.info("***** Start Active Pre-train *****")
    pretrain_resumed = False
    selected_frames_list = []

    backbone_init_ckpt = str(backbone_dir / 'init_checkpoint.pth')

    if not os.path.isfile(backbone_init_ckpt):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
        torch.save(model_state, backbone_init_ckpt)
        logger.info("**init backbone weights saved...**")

    if cfg.ACTIVE_TRAIN.TRAIN_RESUME:
        # try:
        backbone_ckpt_list = [i for i in glob.glob(str(backbone_dir / 'checkpoint_epoch_*.pth'))]
        assert(len(backbone_ckpt_list) > 0) # otherwise nothing to resume
        ckpt_list = [i for i in glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))]

        # filter out the backbones trained after active_pre_train_epochs
        backbone_ckpt_list = [i for i in backbone_ckpt_list if int(i.split('_')[-1].split('.')[0]) <= active_pre_train_epochs]
        if len(ckpt_list) < 1:
            # just at the pretrain stage
        
            backbone_ckpt_list.sort(key=os.path.getmtime)
            last_epoch = int(backbone_ckpt_list[-1].split('_')[-1].split('.')[0])
            if last_epoch >= active_pre_train_epochs:
                pretrain_resumed = True
            elif cfg.ACTIVE_TRAIN.METHOD=='llal':
                logger.info("need to finish the backbone pre-training first...")
                raise NotImplementedError
            logger.info('found {}th epoch pretrain model weights, start resumeing...'.format(last_epoch))
            model_str = str(backbone_dir / backbone_ckpt_list[-1])
            ckpt_last_epoch = torch.load(str(backbone_dir / backbone_ckpt_list[-1]))
        
        else:
            # at the active train stage
            pretrain_resumed = True
            ckpt_list.sort(key=os.path.getmtime)
            last_epoch = int(ckpt_list[-1].split('_')[-1].split('.')[0])
            model_str = str(ckpt_save_dir / ckpt_list[-1])
            ckpt_last_epoch = torch.load(str(ckpt_save_dir / ckpt_list[-1]))

            selected_frames_list = [str(active_label_dir / i) for i in glob.glob(str(active_label_dir / 'selected_frames_epoch_*.pkl'))]
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
        else:
            model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
        # optimizer.load_state_dict(ckpt_last_epoch['optimizer_state'])
        if ckpt_last_epoch['lr_scheduler'] is not None:

            lr_scheduler.load_state_dict(ckpt_last_epoch['lr_scheduler']) 
        
        start_epoch = int(model_str.split('_')[-1].split('.')[0])
        cur_epoch = start_epoch
        accumulated_iter = ckpt_last_epoch['it']
        if len(selected_frames_list) > 0:
            labelled_loader, unlabelled_loader = resume_datset(labelled_loader, unlabelled_loader, selected_frames_list, dist_train, logger, cfg)
            trained_steps = (cur_epoch - cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS) % cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL
            new_accumulated_iter = len(labelled_loader) * trained_steps

        # except Exception:
        #     cfg.ACTIVE_TRAIN.TRAIN_RESUME = False
        #     logger.info('no backbone found. training from strach.')
        #     pass
        
        
    if not pretrain_resumed:
      
        with tqdm.trange(start_epoch, active_pre_train_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
            total_it_each_epoch = len(labelled_loader)
            
            if merge_all_iters_to_one_epoch:
                assert hasattr(labelled_loader.dataset, 'merge_all_iters_to_one_epoch')
                labelled_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=active_pre_train_epochs)
                total_it_each_epoch = len(labelled_loader) // max(active_pre_train_epochs, 1)

            dataloader_iter = iter(labelled_loader)
            for cur_epoch in tbar:
                if labelled_sampler is not None:
                    labelled_sampler.set_epoch(cur_epoch)
                # train one epoch

                
                if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                    cur_scheduler = lr_warmup_scheduler # TODO: currently not related 
                else:
                    cur_scheduler = lr_scheduler
                accumulated_iter = train_one_epoch(
                    model, optimizer, labelled_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == active_pre_train_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter
                )
                # save pre-trained model
                trained_epoch = cur_epoch + 1
                if trained_epoch > 19 and trained_epoch % ckpt_save_interval == 0 and rank == 0:
                    backbone_ckpt_list = glob.glob(str(backbone_dir / 'checkpoint_epoch_*.pth'))
                    backbone_ckpt_list.sort(key=os.path.getmtime)
                    if backbone_ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(backbone_ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(backbone_ckpt_list[cur_file_idx])
                    ckpt_name = backbone_dir / ('checkpoint_epoch_%d' % trained_epoch)

                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, lr_scheduler), filename=ckpt_name,
                    )
        start_epoch = active_pre_train_epochs
    
    
    

    
    logger.info("***** Complete Active Pre-train *****")


    # 2. Compute total epochs
    # (Total budgets / selection number each time) * number of epochs needed for each selection
    
    total_epochs = active_pre_train_epochs + \
                   int((cfg.ACTIVE_TRAIN.TOTAL_BUDGET_NUMS / cfg.ACTIVE_TRAIN.SELECT_NUMS) # 50 + (300 / 50) * 10
                       * (cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL))

    # 3 & 4. Active training loop
    logger.info("***** Start Active Train Loop *****")
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True,
                     leave=(rank == 0)) as tbar:

        #
        # if merge_all_iters_to_one_epoch:
        #     assert hasattr(unlabelled_loader.dataset, 'merge_all_iters_to_one_epoch')
        #     unlabelled_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
        #     total_it_each_epoch = len(unlabelled_loader) // max(total_epochs, 1)
        
        selection_num = 0

        for cur_epoch in tbar:

            # 1) query, select and move active labelled samples
            if cur_epoch == active_pre_train_epochs or \
                    ((cur_epoch - active_pre_train_epochs)
                     % cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL == 0):
                
                    # train loss_net for llal
                if cfg.ACTIVE_TRAIN.METHOD=='llal':
                    if os.path.isfile(os.path.join(ckpt_save_dir, 'init_loss_net_checkpoint.pth')):
                        loss_net_ckpt = torch.load(os.path.join(ckpt_save_dir, 'pretrained_loss_net_{}.pth'.format(cur_epoch)))
                        model.roi_head.loss_net.load_state_dict(loss_net_ckpt['model_state'])
                    else:
                        ckpt_name = os.path.join(ckpt_save_dir, 'init_loss_net_checkpoint.pth')

                        save_checkpoint(
                            checkpoint_state(model.roi_head.loss_net, optimizer=None, epoch=cur_epoch, it=accumulated_iter, lr_scheduler=None), filename=ckpt_name,
                        )
                    model.train()

                    logger.info('**Epoch {}**: start training the loss net...'.format(cur_epoch))
                    with tqdm.trange(0, cfg.ACTIVE_TRAIN.LOSS_NET_TRAIN_EPOCH, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
                        total_it_each_epoch = len(labelled_loader)
                        
                        if merge_all_iters_to_one_epoch:
                            assert hasattr(labelled_loader.dataset, 'merge_all_iters_to_one_epoch')
                            labelled_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=active_pre_train_epochs)
                            total_it_each_epoch = len(labelled_loader) // max(active_pre_train_epochs, 1)

                        dataloader_iter = iter(labelled_loader)

                        # train model's loss net only
                        for param in model.roi_head.loss_net.parameters():
                            param.requires_grad = True
                        loss_net_optimizer = build_optimizer(model.roi_head.loss_net, cfg.OPTIMIZATION)
                        loss_net_scheduler, _ = build_scheduler(loss_net_optimizer, total_iters_each_epoch=total_it_each_epoch, total_epochs=cfg.ACTIVE_TRAIN.LOSS_NET_TRAIN_EPOCH, last_epoch=None, optim_cfg=cfg.OPTIMIZATION)
                        for temp_cur_epoch in tbar:
                            if labelled_sampler is not None:
                                labelled_sampler.set_epoch(temp_cur_epoch)
                            # train one epoch
                            
                            accumulated_iter = train_one_epoch(
                                model, loss_net_optimizer, labelled_loader, model_func,
                                lr_scheduler=loss_net_scheduler,
                                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                                rank=rank, tbar=tbar, tb_log=tb_log,
                                # leave_pbar=(cur_epoch + 1 == cfg.ACTIVE_TRAIN.LOSS_NET_TRAIN_EPOCH),
                                total_it_each_epoch=total_it_each_epoch,
                                dataloader_iter=dataloader_iter
                            )
                        # save pre-trained model
                                
                        if rank == 0:
                            
                            ckpt_name = os.path.join(ckpt_save_dir, 'pretrained_loss_net_{}'.format(cur_epoch))

                            save_checkpoint(
                                checkpoint_state(model.roi_head.loss_net, optimizer=None, epoch=cur_epoch, it=accumulated_iter, lr_scheduler=None), filename=ckpt_name,
                            )

                        # prevent loss net to be trained with other parts
                        for param in model.roi_head.loss_net.parameters():
                                param.requires_grad = False

                selection_num = selection_num + cfg.ACTIVE_TRAIN.SELECT_NUMS
                # select new active samples
                # unlabelled_loader.dataset.eval()
                labelled_loader, unlabelled_loader \
                    = active_training_utils.select_active_labels(
                    model,
                    labelled_loader,
                    unlabelled_loader,
                    rank,
                    logger,
                    method = cfg.ACTIVE_TRAIN.METHOD,
                    leave_pbar=True,
                    cur_epoch=cur_epoch,
                    dist_train=dist_train,
                    active_label_dir=active_label_dir,
                    accumulated_iter=accumulated_iter
                )
                # reset the lr of optimizer and lr scheduler after each selection round
                total_iters_each_epoch = len(labelled_loader)
                # decay_rate = cur_epoch // cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL + 2

                # training from scratch
                logger.info("**finished selection: reload init weights of the model")
                backbone_init_ckpt = torch.load(str(backbone_dir / 'init_checkpoint.pth'))
                model.load_state_dict(backbone_init_ckpt, strict=cfg.ACTIVE_TRAIN.METHOD!='llal') # strict=cfg.ACTIVE_TRAIN.METHOD!='llal'

                # rebuild optimizer and scheduler
                for g in optimizer.param_groups:
                    g['lr'] = cfg.OPTIMIZATION.LR #/ decay_rate

                lr_scheduler, lr_warmup_scheduler = build_scheduler(
                    optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL,
                    last_epoch=None, optim_cfg=cfg.OPTIMIZATION
                )

                # iter counter to zero
            
                new_accumulated_iter = 1

            if labelled_sampler is not None:
                labelled_sampler.set_epoch(cur_epoch)

            # 2) train one epoch
            # labelled_loader.dataset.train()
            total_it_each_epoch = len(labelled_loader)
            logger.info("currently {} iterations to learn per epoch".format(total_it_each_epoch))
            dataloader_iter = iter(labelled_loader)

            if labelled_sampler is not None:
                labelled_sampler.set_epoch(cur_epoch)
            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler # TODO - currently not related
            else:
                cur_scheduler = lr_scheduler
            new_accumulated_iter, accumulated_iter = train_one_epoch(
                model, optimizer, labelled_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=new_accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                # leave_pbar=(cur_epoch + 1 == active_pre_train_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                history_accumulated_iter=accumulated_iter
            )

            # 3) save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)
                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])
                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % (trained_epoch))
                state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, lr_scheduler)
                save_checkpoint(state, filename=ckpt_name)
                logger.info('***checkpoint {} has been saved***'.format(trained_epoch))


