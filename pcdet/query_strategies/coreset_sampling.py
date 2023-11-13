
import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
import torch.nn.functional as F
import tqdm

class CoresetSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(CoresetSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)


    def pairwise_squared_distances(self, x, y):
        # x: N * D * D * 1
        # y: M * D * D * 1
        # return: N * M    
        assert (len(x.shape)) > 1
        assert (len(y.shape)) > 1  
        n = x.shape[0]
        m = y.shape[0]
        x = x.view(n, -1)
        y = y.view(m, -1)
                                                              
        x_norm = (x**2).sum(1).view(n, 1)
        y_t = y.permute(1, 0).contiguous()
        y_norm = (y**2).sum(1).view(1, m)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist[dist != dist] = 0 # replace nan values with 0
        return torch.clamp(dist, 0.0, float('inf'))

    def furthest_first(self, X, X_set, n):
        m = X.shape[0]
        m_label = X_set.shape[0]
        X = X.view(m, -1)
        X_set = X_set.view(m_label, -1)
        # dist_ctr: num_unlabel * num_label
        dist_ctr = self.pairwise_squared_distances(X, X_set)
        # min_dist: [num_unlabel]-D -> record the nearest distance to the orig X_set
        min_dist = dist_ctr.mean(1)
        idxs = []
        for i in range(n):
            # choose the furthest to orig X_set
            idx = torch.argmax(min_dist)
            idxs.append(idx)
            # no need to calculate if the last iter
            if i < (n-1):
                dist_new_ctr = self.pairwise_squared_distances(X, X[idx, :].unsqueeze(0))
                for j in range(m):
                    min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
        return idxs
            


    def query(self, leave_pbar=True, cur_epoch=None):
        select_dic = {}

        
        self.model.eval()
        unlabeled_embeddings = []
        labeled_embeddings = []
        unlabeled_frame_ids = []

        # feed unlabeled data forward the model
        total_it_each_epoch = len(self.unlabelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
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
                    self.frame_recall[unlabelled_batch['frame_id'][batch_inx]] = _
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    unlabeled_frame_ids.append(unlabelled_batch['frame_id'][batch_inx])
                embeddings = pred_dicts[0]['embeddings'].view(-1, 128, self.cfg.MODEL.ROI_HEAD.SHARED_FC[-1])
                unlabeled_embeddings.append(embeddings)
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        
        # feed labeled data forward the model
        total_it_each_epoch = len(self.labelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_labelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
        val_dataloader_iter = iter(self.labelled_loader)
        val_loader = self.labelled_loader
        for cur_it in range(total_it_each_epoch):
            try:
                labelled_batch = next(val_dataloader_iter)
            except StopIteration:
                labelled_dataloader_iter = iter(val_loader)
                labelled_batch = next(labelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(labelled_batch)
                pred_dicts, _ = self.model(labelled_batch)

                batch_size = len(pred_dicts)
                    
                embeddings = pred_dicts[0]['embeddings'].view(-1, 128, self.cfg.MODEL.ROI_HEAD.SHARED_FC[-1])
                labeled_embeddings.append(embeddings)
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

       
        print('** [Coreset] start searching...**')
        selected_idx = self.furthest_first(torch.cat(unlabeled_embeddings, 0), torch.cat(labeled_embeddings, 0), n=self.cfg.ACTIVE_TRAIN.SELECT_NUMS)
        selected_frames = [unlabeled_frame_ids[idx] for idx in selected_idx]
        self.report_open_world_recall(self.frame_recall, selected_frames)
        return selected_frames
