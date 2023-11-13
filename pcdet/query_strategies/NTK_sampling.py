
import torch
import torch.nn as nn
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
import torch.nn.functional as F
import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'bmdal_reg')))
sys.path.append(os.path.abspath(os.path.join('..', 'bmdal_reg/bmdal')))
from .bmdal_reg.bmdal.feature_data import TensorFeatureData
from .bmdal_reg.bmdal.algorithms import select_batch
from .bmdal_reg.models import create_tabular_model
from torch.distributions import Categorical


class NTKSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(NTKSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)
        self.k = 3
        self.kernel_transform = {'maxdet':  [('rp', [cfg.ACTIVE_TRAIN.LATENT_DIM]), ('train', [0.1, None])], 'bait': [('rp', [cfg.ACTIVE_TRAIN.LATENT_DIM]), ('train', [0.1, None])]}

    def query(self, leave_pbar=True, cur_epoch=None):
        select_dic = {}

        unlabeled_embeddings = []
        labeled_embeddings = []
        unlabeled_frame_ids = []

                # feed labeled data forward the model

        total_it_each_epoch = len(self.labelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_labelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
        val_dataloader_iter = iter(self.labelled_loader)
        val_loader = self.labelled_loader
        reg_train = []
        cls_train = []
        for cur_it in range(total_it_each_epoch):
            try:
                labelled_batch = next(val_dataloader_iter)
            except StopIteration:
                labelled_dataloader_iter = iter(val_loader)
                labelled_batch = next(labelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(labelled_batch)
                self.model.train()
                train_pred_dicts, _, _= self.model(labelled_batch)
                if self.cfg.MODEL.NAME == 'PVRCNN':
                    reg_train.append(train_pred_dicts['rcnn_reg_gt'].view(-1, 128, 7))
                cls_train.append(train_pred_dicts['rcnn_cls_gt'])
                
                self.model.eval()
                eval_pred_dicts, _ = self.model(labelled_batch)

                batch_size = len(train_pred_dicts)
                # print(eval_pred_dicts[0]['embeddings'].shape)
                embeddings = eval_pred_dicts[0]['embeddings'].view(-1, 128, self.cfg.MODEL.ROI_HEAD.SHARED_FC[-1])
                labeled_embeddings.append(embeddings)
                
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
        
        num_class = len(self.labelled_loader.dataset.class_names)
        # feed unlabeled data forward the model
        total_it_each_epoch = len(self.unlabelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader

        entropy_list = []
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)
                batch_size = len(pred_dicts)
                for batch_inx in range(len(pred_dicts)):
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    unlabeled_frame_ids.append(unlabelled_batch['frame_id'][batch_inx])

                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = torch.tensor(0).cuda()
                    else:
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        entropy = Categorical(probs = unique_proportions / sum(counts)).entropy()

                    entropy_list.append(entropy)

                embeddings = pred_dicts[0]['embeddings'].view(-1, 128, self.cfg.MODEL.ROI_HEAD.SHARED_FC[-1])
                unlabeled_embeddings.append(embeddings)
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        # entropy_list = torch.stack(entropy_list)
        # _, preselected_idx = torch.sort(entropy_list, stable=True, descending=True)
        # preselected_idx = preselected_idx[:self.k * self.cfg.ACTIVE_TRAIN.SELECT_NUMS]

        # unlabeled_frame_ids = [unlabeled_frame_ids[i] for i in preselected_idx]
        print('** NTK start searching...**')# 100 * 32768
        labeled_embeddings = torch.cat(labeled_embeddings, 0)
        labeled_embeddings = labeled_embeddings.view(labeled_embeddings.shape[0], -1)
        unlabeled_embeddings = torch.cat(unlabeled_embeddings, 0)
        unlabeled_embeddings = unlabeled_embeddings.view(unlabeled_embeddings.shape[0], -1)

        # unlabeled_embeddings = unlabeled_embeddings[unlabeled_frame_ids, :]
        # custom_model = nn.Sequential(nn.Linear(, 1000), nn.SiLU(), nn.Linear(1000, 128*7)).cuda()
        if self.cfg.MODEL.NAME == 'PVRCNN':
            y_train = torch.cat((torch.cat(reg_train, 0), torch.cat(cls_train, 0).unsqueeze(-1)), -1)
            y_train = y_train.view(y_train.shape[0], -1)
        else:
            y_train = torch.cat(cls_train, 0)
        
        custom_model = nn.Sequential(nn.Linear(labeled_embeddings.shape[-1], self.cfg.ACTIVE_TRAIN.LATENT_DIM), nn.SiLU(), nn.Linear(self.cfg.ACTIVE_TRAIN.LATENT_DIM, 100), nn.SiLU(), nn.Linear(100, y_train.shape[-1])).cuda()
        opt = torch.optim.Adam(custom_model.parameters(), lr=2e-2)
        for epoch in range(10):
            y_pred = custom_model(labeled_embeddings)
            loss = ((y_pred - y_train)**2).mean()
            train_rmse = loss.sqrt().item()
            # pool_rmse = ((custom_model(x_pool) - y_pool)**2).mean().sqrt().item()
            print(f'train RMSE: {train_rmse:5.3f}')
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        train_data = TensorFeatureData(labeled_embeddings)
        pool_data = TensorFeatureData(unlabeled_embeddings)
        selected_idx, _ = select_batch(batch_size=self.cfg.ACTIVE_TRAIN.SELECT_NUMS, models=[custom_model], 
                                data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                                selection_method=self.cfg.ACTIVE_TRAIN.SELECTION_METHOD,
                                base_kernel=self.cfg.ACTIVE_TRAIN.BASE_KERNEL, kernel_transforms=self.kernel_transform[self.cfg.ACTIVE_TRAIN.SELECTION_METHOD],
                                entropy_list=entropy_list, maxdet_sigma=0.1, allow_float64=True, bait_sigma=0.1, lr=0.375, 
                                weight_gain=0.2, bias_gain=0.2, entropy_sigma=self.cfg.ACTIVE_TRAIN.ENT_SIGMA)

        
        selected_frames = [unlabeled_frame_ids[idx] for idx in selected_idx]

        return selected_frames
