import os
import pickle
import wandb
import torch
class Strategy:
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        
        self.cfg = cfg
        self.active_label_dir = active_label_dir
        self.rank = rank
        self.model = model
        self.labelled_loader = labelled_loader
        self.unlabelled_loader = unlabelled_loader
        self.labelled_set = labelled_loader.dataset
        self.unlabelled_set = unlabelled_loader.dataset
        self.bbox_records = {}
        self.point_measures = ['mean', 'median', 'variance']
        for met in self.point_measures:
            setattr(self, '{}_point_records'.format(met), {})
   


        if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
            self.pairs = list(zip(self.unlabelled_set.sample_id_list, self.unlabelled_set.kitti_infos))
        else:
            self.pairs = list(zip(self.unlabelled_set.frame_ids, self.unlabelled_set.infos))

    def save_points(self, frame_id, batch_dict):
        # 'num_bbox': num_bbox,
        # 'mean_points': mean_points,
        # 'median_points': median_points,
        # 'variance_points': variance_points,

        self.bbox_records[frame_id] = batch_dict['num_bbox']
        
        self.mean_point_records[frame_id] = batch_dict['mean_points']
        self.median_point_records[frame_id] = batch_dict['median_points']
        self.variance_point_records[frame_id] = batch_dict['variance_points']
           
    

    def update_dashboard(self, cur_epoch=None, accumulated_iter=None):

        classes = list(self.selected_bbox[0].keys())
    
    
        total_bbox = 0
        for cls_idx in classes:
            
            num_cls_bbox = sum([i[cls_idx] for i in self.selected_bbox])
            wandb.log({'active_selection/num_bbox_{}'.format(cls_idx): num_cls_bbox}, step=accumulated_iter)
            total_bbox += num_cls_bbox
        
            if num_cls_bbox:
                for met in self.point_measures:
                    stats_point = sum([i[cls_idx] for i in getattr(self, 'selected_{}_points'.format(met))]) / len(getattr(self, 'selected_{}_points'.format(met)))
                    wandb.log({'active_selection/{}_points_{}'.format(met, cls_idx): stats_point}, step=accumulated_iter)
            else:
                for met in self.point_measures:
                    wandb.log({'active_selection/{}_points_{}'.format(met, cls_idx): 0}, step=accumulated_iter)

        
        wandb.log({'active_selection/total_bbox_selected': total_bbox}, step=accumulated_iter)

    
    def save_active_labels(self, selected_frames=None, grad_embeddings=None, cur_epoch=None):
      
        if selected_frames is not None:
            self.selected_bbox = [self.bbox_records[i] for i in selected_frames]
            for met in self.point_measures:
                setattr(self, 'selected_{}_points'.format(met), [getattr(self, '{}_point_records'.format(met))[i] for i in selected_frames])

            with open(os.path.join(self.active_label_dir, 'selected_frames_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
                pickle.dump({'frame_id': selected_frames, 'selected_mean_points': self.selected_mean_points, 'selected_bbox': self.selected_bbox, \
                'selected_median_points': self.selected_median_points, 'selected_variance_points': self.selected_variance_points}, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))

        if grad_embeddings is not None:
            with open(os.path.join(self.active_label_dir, 'grad_embeddings_epoch_{}.pkl'.format(cur_epoch)), 'wb') as f:
                pickle.dump(grad_embeddings, f)
            print('successfully saved grad embeddings for epoch {}'.format(cur_epoch))

    def query(self, leave_pbar=True, cur_epoch=None):
        pass