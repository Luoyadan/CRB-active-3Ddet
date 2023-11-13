
import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
from pcdet.datasets import build_active_dataloader
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm
import numpy as np
import wandb
import time
import scipy
from sklearn.cluster import kmeans_plusplus, KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform
from sklearn.neighbors import KernelDensity
from scipy.cluster.vq import vq
from typing import Dict, List
from pcdet.config import cfg

class GradnormSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg, cur_epoch=None):

        super(GradnormSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

        # coefficients controls the ratio of selected subset
        
        # bandwidth for the KDE in the GPDB module
        # controls the boundary of the uniform prior distribution
        self.alpha = 0.95
        self.ow_epoch = False
        if cur_epoch == cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS:
            self.ow_epoch = True

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))

    def query(self, leave_pbar=True, cur_epoch=None):
        select_dic = {}

        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS

        self.model.eval()
        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
        total_it_each_epoch = len(self.unlabelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='feed_forward_%d' % cur_epoch, dynamic_ncols=True)
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)

                for batch_inx in range(len(pred_dicts)):
                    self.frame_recall[unlabelled_batch['frame_id'][batch_inx]] = _
                    self.save_points(unlabelled_batch['frame_id'][batch_inx],
                                     pred_dicts[batch_inx])
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()



        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
        total_it_each_epoch = len(self.unlabelled_loader)
        # grad norm the model
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='extracting_grad_norm_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        # self.enable_dropout(self.model)
        check_value = []

        self.model.train()
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)

            load_data_to_gpu(unlabelled_batch)
            _  = self.model(unlabelled_batch)
            train_dict = _[0]
            # https://github.com/KingJamesSong/RankFeat/blob/2fc44fca4ffea0dac23be16bf8030f7e48642dc3/test_ood_resnetv2.py#L135C4-L135C4
            # get sample wise grads
            rcnn_cls_preds_per_sample = train_dict['rcnn_cls']
            targets = torch.ones((rcnn_cls_preds_per_sample.shape[0]), 1).cuda()
            # rcnn_reg_preds_per_sample = pred_dicts['rcnn_reg'] if 'rcnn_reg' in pred_dicts.keys() else None

            # if rcnn_reg_preds_per_sample is not None:  # pvrcnn
            #     cls_loss, _ = self.model.roi_head.get_box_cls_layer_loss(
            #         {'rcnn_cls': rcnn_cls_preds_per_sample,
            #          'rcnn_cls_labels': rcnn_cls_gt_per_sample})
            #     reg_loss = self.model.roi_head.get_box_reg_layer_loss(
            #         {'rcnn_reg': rcnn_reg_preds_per_sample,
            #          'reg_sample_targets': rcnn_reg_gt_per_sample})

            # else:  # second_iou
            cls_loss, _ = self.model.roi_head.get_box_iou_layer_loss(
                {'rcnn_cls': rcnn_cls_preds_per_sample,
                 'rcnn_cls_labels': targets})

            del rcnn_cls_preds_per_sample, targets

            torch.cuda.empty_cache()

            self.model.zero_grad()
            cls_loss.backward()

            fc_grads = self.model.roi_head.shared_fc_layer[4].weight.grad.clone().detach().cpu()
            layer_grad_norm = torch.sum(torch.abs(fc_grads)).cpu().numpy()
            # fc_grad_norm_list.append(fc_grads)
            # index_list.append(unlabelled_batch['frame_id'][0])
            select_dic[unlabelled_batch['frame_id'][0]] = layer_grad_norm
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        check_value.sort()
        log_data = [[idx, value] for idx, value in enumerate(check_value)]

        # sort and get selected_frames
        select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        # narrow down the scope
        selected_frames = list(select_dic.keys())[:int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS)]
        self.model.eval()
        # returned the index of acquired bounding boxes
        self.report_open_world_recall(self.frame_recall, selected_frames)
        return selected_frames
        selected_id_list, selected_infos = [], []
        unselected_id_list, unselected_infos = [], []

        '''
        -------------  Stage 2: Representative Prototype Selection ----------------------
        '''

        # rebuild a dataloader for K1 samples
        for i in range(len(self.pairs)):
            if self.pairs[i][0] in selected_frames:
                selected_id_list.append(self.pairs[i][0])
                selected_infos.append(self.pairs[i][1])
            else:
                # no need for unselected part
                if len(unselected_id_list) == 0:
                    unselected_id_list.append(self.pairs[i][0])
                    unselected_infos.append(self.pairs[i][1])

        selected_id_list, selected_infos, \
        unselected_id_list, unselected_infos = \
            tuple(selected_id_list), tuple(selected_infos), \
            tuple(unselected_id_list), tuple(unselected_infos)
        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        labelled_set, _, \
        grad_loader, _, \
        _, _ = build_active_dataloader(
            self.cfg.DATA_CONFIG,
            self.cfg.CLASS_NAMES,
            1,
            False,
            workers=self.labelled_loader.num_workers,
            logger=None,
            training=True,
            active_training=active_training
        )
        grad_dataloader_iter = iter(grad_loader)
        total_it_each_epoch = len(grad_loader)

        self.model.train()
        fc_grad_embedding_list = []
        index_list = []


        # start looping over the K1 samples        
        if self.rank == 0:
                pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                                desc='inf_grads_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(grad_dataloader_iter)
                
            except StopIteration:
                unlabelled_dataloader_iter = iter(grad_loader)
                unlabelled_batch = next(grad_dataloader_iter)

            load_data_to_gpu(unlabelled_batch)
                
            pred_dicts, _, _= self.model(unlabelled_batch)

            # get sample wise grads
            rcnn_cls_preds_per_sample = pred_dicts['rcnn_cls']
            rcnn_cls_gt_per_sample = cls_results[unlabelled_batch['frame_id'][0]]

            rcnn_reg_preds_per_sample = pred_dicts['rcnn_reg'] if 'rcnn_reg' in pred_dicts.keys() else None
            rcnn_reg_gt_per_sample = reg_results[unlabelled_batch['frame_id'][0]]

            if rcnn_reg_preds_per_sample is not None: #pvrcnn
                cls_loss, _ = self.model.roi_head.get_box_cls_layer_loss(
                    {'rcnn_cls': rcnn_cls_preds_per_sample,
                     'rcnn_cls_labels': rcnn_cls_gt_per_sample})
                reg_loss = self.model.roi_head.get_box_reg_layer_loss(
                    {'rcnn_reg': rcnn_reg_preds_per_sample,
                     'reg_sample_targets': rcnn_reg_gt_per_sample})

            else: # second_iou
                cls_loss, _ = self.model.roi_head.get_box_iou_layer_loss(
                    {'rcnn_cls': rcnn_cls_preds_per_sample,
                     'rcnn_cls_labels': rcnn_cls_gt_per_sample})


            # clean cache
            del rcnn_cls_preds_per_sample, rcnn_cls_gt_per_sample
            if rcnn_reg_preds_per_sample is not None:
                del rcnn_reg_preds_per_sample, rcnn_reg_gt_per_sample

            torch.cuda.empty_cache()

            loss = cls_loss + reg_loss.mean() if rcnn_reg_preds_per_sample is not None else cls_loss
            self.model.zero_grad()
            loss.backward()

            fc_grads = self.model.roi_head.shared_fc_layer[4].weight.grad.clone().detach().cpu()
            fc_grad_embedding_list.append(fc_grads)
            index_list.append(unlabelled_batch['frame_id'][0])

            if self.rank == 0:
                pbar.update()
                    # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        # stacking gradients for K1 candiates
        fc_grad_embeddings = torch.stack(fc_grad_embedding_list, 0)
        num_sample = fc_grad_embeddings.shape[0]
        fc_grad_embeddings = fc_grad_embeddings.view(num_sample, -1)
        start_time = time.time()

        # choose the prefered prototype selection method and select the K2 medoids
        if not cfg.ACTIVE_TRAIN.ENABLE_S3:
            self.k2=1

        if self.prototype == 'kmeans++':
            _, selected_fc_idx = kmeans_plusplus(fc_grad_embeddings.numpy(), n_clusters=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2), random_state=0)
        elif self.prototype == 'kmeans':
            km = KMeans(n_clusters=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2), random_state=0).fit(fc_grad_embeddings.numpy())
            selected_fc_idx, _ = vq(km.cluster_centers_, fc_grad_embeddings.numpy())
        elif self.prototype == 'birch':
            ms = Birch(n_clusters=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2)).fit(fc_grad_embeddings.numpy())
            selected_fc_idx, _ = vq(ms.subcluster_centers_, fc_grad_embeddings.numpy())
        elif self.prototype == 'gmm':
            gmm = GaussianMixture(n_components=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2), random_state=0, covariance_type="diag").fit(fc_grad_embeddings.numpy())
            selected_fc_idx, _ = vq(gmm.means_, fc_grad_embeddings.numpy())
        else:
            raise NotImplementedError
        selected_frames = [index_list[i] for i in selected_fc_idx]
        print("--- {%s} running time: %s seconds for fc grads---" % (self.prototype, time.time() - start_time))

        if not cfg.ACTIVE_TRAIN.ENABLE_S3:
            self.model.eval()
            # returned the index of acquired bounding boxes
            self.report_open_world_recall(self.frame_recall, selected_frames)
            return selected_frames

        '''
        -------------  Stage 3: Greedy Point Cloud Density Balancing ----------------------
        '''


        sampled_density_list = [density_list[i] for i in selected_frames]
        sampled_label_list = [label_list[i] for i in selected_frames]

        """ Build the uniform distribution for each class """
        start_time = time.time()
        density_all = torch.cat(list(density_list.values()), 0)
        label_all = torch.cat(list(label_list.values()), 0)
        num_class = len(label_all.unique()) if len(label_all.unique()) != num_class else num_class
        unique_labels, label_counts = torch.unique(label_all, return_counts=True)
        sorted_density = [torch.sort(density_all[label_all==unique_label])[0] for unique_label in unique_labels]
        global_density_max = [int(sorted_density[unique_label][-1]) for unique_label in range(len(unique_labels))]
        global_density_high = [int(sorted_density[unique_label][int(self.alpha * label_counts[unique_label])]) for unique_label in range(len(unique_labels))]
        global_density_low = [int(sorted_density[unique_label][-int(self.alpha * label_counts[unique_label])]) for unique_label in range(len(unique_labels))]
        x_axis = [np.linspace(-50, int(global_density_max[i])+50, 400) for i in range(num_class)]
        uniform_dist_per_cls = [uniform.pdf(x_axis[i], global_density_low[i], global_density_high[i] - global_density_low[i]) for i in range(num_class)]

        print("--- Build the uniform distribution running time: %s seconds ---" % (time.time() - start_time))

        density_list, label_list, frame_id_list = sampled_density_list, sampled_label_list, selected_frames

        selected_frames: List[str] = []
        selected_box_densities: torch.tensor = torch.tensor([]).cuda()
        selected_box_labels: torch.tensor = torch.tensor([]).cuda()


        # looping over N_r samples
        if self.rank == 0:
            pbar = tqdm.tqdm(total=self.cfg.ACTIVE_TRAIN.SELECT_NUMS, leave=leave_pbar,
                             desc='global_density_div_for_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
        for j in range(self.cfg.ACTIVE_TRAIN.SELECT_NUMS):
            if j == 0: # initially, we randomly select a frame.

                selected_frames.append(frame_id_list[j])
                selected_box_densities = torch.cat((selected_box_densities, density_list[j]))
                selected_box_labels = torch.cat((selected_box_labels, label_list[j]))

                # remove selected frame
                del density_list[0]
                del label_list[0]
                del frame_id_list[0]

            else: # go through all the samples and choose the frame that can most reduce the KL divergence
                best_frame_id = None
                best_frame_index = None
                best_inverse_coff = -1

                for i in range(len(density_list)):
                    unique_proportions = np.zeros(num_class)
                    KL_scores_per_cls = np.zeros(num_class)

                    for cls in range(num_class):
                        if (label_list[i] == cls + 1).sum() == 0:
                            unique_proportions[cls] = 1
                            KL_scores_per_cls[cls] = np.inf
                        else:
                            # get existing selected box densities
                            selected_box_densities_cls = selected_box_densities[selected_box_labels==(cls + 1)]
                            # append new frame's box densities to existing one
                            selected_box_densities_cls = torch.cat((selected_box_densities_cls,
                                                                    density_list[i][label_list[i] == (cls + 1)]))
                            # initialize kde
                            kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(
                                selected_box_densities_cls.cpu().numpy()[:, None])

                            logprob = kde.score_samples(x_axis[cls][:, None])
                            KL_score_per_cls = scipy.stats.entropy(uniform_dist_per_cls[cls], np.exp(logprob))
                            KL_scores_per_cls[cls] = KL_score_per_cls
                            # ranging from 0 to 1
                            unique_proportions[cls] = 2 / np.pi * np.arctan(np.pi / 2 * KL_score_per_cls)

                    inverse_coff = np.mean(1 - unique_proportions)
                    # KL_save_list.append(inverse_coff)
                    if inverse_coff > best_inverse_coff:
                        best_inverse_coff = inverse_coff
                        best_frame_index = i
                        best_frame_id = frame_id_list[i]

                # remove selected frame
                selected_box_densities = torch.cat((selected_box_densities, density_list[best_frame_index]))
                selected_box_labels = torch.cat((selected_box_labels, label_list[best_frame_index]))
                del density_list[best_frame_index]
                del label_list[best_frame_index]
                del frame_id_list[best_frame_index]

                selected_frames.append(best_frame_id)

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        self.model.eval()
        # returned the index of acquired bounding boxes
        self.report_open_world_recall(self.frame_recall, selected_frames)
        return selected_frames
