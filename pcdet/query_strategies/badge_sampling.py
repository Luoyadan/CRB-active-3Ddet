
import itertools
import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
import torch.nn.functional as F
import tqdm
from sklearn.cluster import kmeans_plusplus
import time
import pickle
import os
from torch.utils.data import DataLoader
import itertools
from collections import Counter
import gc
import numpy as np

class BadgeSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(BadgeSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)
        # num_box decay rate
        # self.delta = 0.99
        # self.k = 3
    
    def get_grad_embedding(self, probs, feat):
        embeddings_pos = feat * (1 - probs)

        embeddings = torch.cat((embeddings_pos, -embeddings_pos), axis=-1)
        final_embeddings = torch.clone(embeddings)
        final_embeddings[probs < 0.5] = torch.cat((-embeddings_pos[probs < 0.5], embeddings_pos[probs < 0.5]), axis=1)
        # B x 1 true false true

        return embeddings

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
        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
        total_it_each_epoch = len(self.unlabelled_loader)

        # feed forward the model
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        # enable dropout for multiple sampling
        self.enable_dropout(self.model)
        # self.model.roi_head.shared_fc_layer[3].train()
        
        rpn_preds_results = []
        # cls_results = []
        # reg_results = []

        if os.path.isfile(os.path.join(self.active_label_dir, 'grad_embeddings_epoch_{}.pkl'.format(cur_epoch))):
            # if found, then resume
            print('found {} epoch grad embeddings... start resuming...'.format(cur_epoch))
            with open(os.path.join(self.active_label_dir, 'grad_embeddings_epoch_{}.pkl'.format(cur_epoch)), 'rb') as f:
                grad_embeddings = pickle.load(f)
        else:

            for cur_it in range(total_it_each_epoch):
                try:
                    unlabelled_batch = next(val_dataloader_iter)
                except StopIteration:
                    unlabelled_dataloader_iter = iter(val_loader)
                    unlabelled_batch = next(unlabelled_dataloader_iter)
                with torch.no_grad():
                    
                    load_data_to_gpu(unlabelled_batch)
                
                    pred_dicts, _= self.model(unlabelled_batch)
                    for batch_inx in range(len(pred_dicts)):
                        self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                        # final_full_cls_logits = pred_dicts[batch_inx]['pred_logits']
                    # did not apply batch mask -> directly output 
                    rpn_preds = pred_dicts[0]['rpn_preds']
                    batch_size = rpn_preds.shape[0]
                    rpn_preds = torch.argmax(rpn_preds.view(batch_size, -1, self.model.dense_head.num_class), -1)
                    rpn_preds_results.append(rpn_preds.cpu())
                    # cls_results.append(torch.mean(torch.sigmoid(pred_dicts[0]['rcnn_cls']), 0).view(batch_size, -1, 1))
                    # reg_results.append(torch.mean(pred_dicts[0]['rcnn_reg'], 0).view(batch_size, -1, 7))
                if self.rank == 0:
                    pbar.update()
                        # pbar.set_postfix(disp_dict)
                    pbar.refresh()
            
            
            if self.rank == 0:
                pbar.close()
            del rpn_preds
            del pred_dicts
            torch.cuda.empty_cache()
            print('start stacking cls and reg results as gt...')
            # cls_results = torch.cat(cls_results, 0)
            # reg_results = torch.cat(reg_results, 0)
            rpn_preds_results = torch.cat(rpn_preds_results, 0)
            

            print('retrieving grads on the training mode...')
            self.model.train()
            # fc_grad_embedding_list = []
            rpn_grad_embedding_list = []
            # preds_num_bbox_list = []

            grad_loader = DataLoader(
                self.unlabelled_set, batch_size=1, pin_memory=True, num_workers=self.unlabelled_loader.num_workers,
                shuffle=False, collate_fn=self.unlabelled_set.collate_batch,
                drop_last=False, sampler=self.unlabelled_loader.sampler, timeout=0
                )
            grad_dataloader_iter = iter(grad_loader)
            total_it_each_epoch = len(grad_loader)

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
     
                # rcnn_cls_preds_per_sample = pred_dicts['rcnn_cls']
                # rcnn_cls_gt_per_sample = cls_results[cur_it, :]

                # rcnn_reg_preds_per_sample = pred_dicts['rcnn_reg']
                # rcnn_reg_gt_per_sample = reg_results[cur_it, :]

               
                # cls_loss, _ = self.model.roi_head.get_box_cls_layer_loss({'rcnn_cls': rcnn_cls_preds_per_sample, 'rcnn_cls_labels': rcnn_cls_gt_per_sample})
                
                # reg_loss = self.model.roi_head.get_box_reg_layer_loss({'rcnn_reg': rcnn_reg_preds_per_sample, 'reg_sample_targets': rcnn_reg_gt_per_sample})
                # del rcnn_reg_preds_per_sample, rcnn_reg_gt_per_sample
                # torch.cuda.empty_cache()
                # assign the rpn hypothetical labels for loss calculation
                
                # self.model.dense_head.forward_red_dict['box_cls_labels'] = rpn_preds_results[cur_it, :].cuda().unsqueeze(0)
                new_data  = {'box_cls_labels': rpn_preds_results[cur_it, :].cuda().unsqueeze(0), 'cls_preds': pred_dicts['rpn_preds']}
                rpn_loss = self.model.dense_head.get_cls_layer_loss(new_data=new_data)[0]
                # since the rpn head does not have dropout, we cannot get MC dropout labels for regression
                loss = rpn_loss
                self.model.zero_grad()
                loss.backward()
                
                # fc_grads = self.model.roi_head.shared_fc_layer[4].weight.grad.clone().detach().cpu()
                # fc_grad_embedding_list.append(fc_grads)

                rpn_grads = self.model.dense_head.conv_cls.weight.grad.clone().detach().cpu()
                rpn_grad_embedding_list.append(rpn_grads)
                
                # preds_num_bbox = (torch.norm(self.model.roi_head.reg_layers[-1].weight.grad, dim=0) > 0).sum()
                # preds_num_bbox_list.append(preds_num_bbox)

                # del fc_grads
                # del rpn_grads
                # torch.cuda.empty_cache()

                if self.rank == 0:
                    pbar.update()
                        # pbar.set_postfix(disp_dict)
                    pbar.refresh()
                # if cur_it > 100:
                #     print('check here')
                #     break
                    

            if self.rank == 0:
                pbar.close()

            
        rpn_grad_embeddings = torch.stack(rpn_grad_embedding_list, 0)
        del rpn_grad_embedding_list
        gc.collect()
        num_sample = rpn_grad_embeddings.shape[0]
        rpn_grad_embeddings = rpn_grad_embeddings.view(num_sample, -1)
        start_time = time.time()
        _, selected_rpn_idx = kmeans_plusplus(rpn_grad_embeddings.numpy(), n_clusters=self.cfg.ACTIVE_TRAIN.SELECT_NUMS, random_state=0)
        print("--- kmeans++ running time: %s seconds for rpn grads---" % (time.time() - start_time))
        
        selected_idx = selected_rpn_idx
        self.model.zero_grad()
        self.model.eval()
        selected_frames = [self.unlabelled_set.sample_id_list[idx] for idx in selected_idx]
        return selected_frames
