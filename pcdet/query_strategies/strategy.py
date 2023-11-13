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
        self.frame_recall = {}
        self.point_measures = ['mean', 'median', 'variance']
        for met in self.point_measures:
            setattr(self, '{}_point_records'.format(met), {})
   


        if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
            self.pairs = list(zip(self.unlabelled_set.sample_id_list, self.unlabelled_set.kitti_infos))
        elif cfg.DATA_CONFIG.DATASET == "NuScenesDataset":
            self.pairs = list(zip(self.unlabelled_set.lidar_paths, self.unlabelled_set.infos))
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

    def report_open_world_recall(self, frame_recall, selected_frames):
        # init keys for selected_recall
        selected_recall = {}
        for k, v in list(frame_recall.values())[0].items():
            selected_recall[k] = 0

        for frame, recall_dict in frame_recall.items():
            if frame in selected_frames:
                for k, v in selected_recall.items():
                    selected_recall[k] = selected_recall[k] + recall_dict[k]
        import wandb
        wandb.log(selected_recall)

    # def open_world_report(self, pred_boxes, gt_boxes, pred_classes, gt_classes):
    #     from pcdet.config import cfg
    #     from ..ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
    #
    #     # OW report dict
    #     all_classes = torch.tensor(range(1, 1 + len(cfg.CLASS_NAMES))).cuda()
    #     known_classes = torch.tensor(
    #         range(1, 1 + len(cfg.OPEN_WORLD.KNOWN_CLASS_NAMES))).cuda()
    #     unk_classes = torch.tensor(
    #         range(1 + len(cfg.OPEN_WORLD.KNOWN_CLASS_NAMES),
    #               1 + len(cfg.CLASS_NAMES))).cuda()
    #
    #     gt_boxes = gt_boxes[
    #         torch.abs(gt_boxes).sum(dim=1) != 0]  # remove empty GT
    #     gt_classes = gt_boxes[:, -1]
    #
    #
    #     iou_th = 0.3
    #     assert gt_classes.shape[0] == gt_boxes.shape[0]
    #     report_status = "active"
    #
    #     ow_report = {}
    #     for cls in all_classes:
    #         gt_cls_mask, pred_cls_mask = gt_classes == cls, pred_classes == cls
    #         gt_boxes_cls, pred_boxes_cls = gt_boxes[gt_cls_mask], pred_boxes[
    #             pred_cls_mask]
    #         iou = boxes_iou3d_gpu(pred_boxes_cls[:, :7], gt_boxes_cls[:, :7])
    #         match_mask = iou > iou_th
    #
    #         # Compute TP, FP, FN, precision, recall, F1
    #         total_pred_count = pred_boxes_cls.shape[0]
    #         total_gt_count = gt_boxes_cls.shape[0]
    #         TP_count = iou[match_mask].shape[0]
    #         FP_count = total_pred_count - TP_count
    #         FN_count = total_gt_count - TP_count
    #         precision = TP_count / (TP_count + FP_count)
    #         recall = TP_count / (TP_count + FN_count)
    #
    #         def f1(precision, recall):
    #             return 2 * precision * recall / (recall + precision)
    #
    #         F1 = f1(precision, recall)
    #         ow_report[cfg.CLASS_NAMES[cls - 1]][
    #             "total_pred_count"] = total_pred_count
    #         ow_report[cfg.CLASS_NAMES[cls - 1]][
    #             "total_gt_count"] = total_gt_count
    #         ow_report[cfg.CLASS_NAMES[cls - 1]]["TP_count"] = TP_count
    #         ow_report[cfg.CLASS_NAMES[cls - 1]]["FP_count"] = FP_count
    #         ow_report[cfg.CLASS_NAMES[cls - 1]]["FN_count"] = FN_count
    #         ow_report[cfg.CLASS_NAMES[cls - 1]]["precision"] = precision
    #         ow_report[cfg.CLASS_NAMES[cls - 1]]["recall"] = recall
    #         ow_report[cfg.CLASS_NAMES[cls - 1]]["F1"] = F1