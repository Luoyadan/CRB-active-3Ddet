import random
from .strategy import Strategy
import tqdm
import torch
from pcdet.models import load_data_to_gpu
import torch.nn.functional as F

class ReactSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(ReactSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

    def query(self, leave_pbar=True, cur_epoch=None):
        if len(self.bbox_records) == 0:
            
            # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
            val_dataloader_iter = iter(self.unlabelled_loader)
            val_loader = self.unlabelled_loader
            total_it_each_epoch = len(self.unlabelled_loader)
            all_frames = []
            self.frame_recall = {}
            select_dic = {}
            # feed forward the model
            if self.rank == 0:
                pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                                desc='going through_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
            self.model.eval()

            for cur_it in range(total_it_each_epoch):
                try:
                    unlabelled_batch = next(val_dataloader_iter)
                except StopIteration:
                    unlabelled_dataloader_iter = iter(val_loader)
                    unlabelled_batch = next(unlabelled_dataloader_iter)
                with torch.no_grad():
                    load_data_to_gpu(unlabelled_batch)
                    pred_dicts, recall_dicts = self.model(unlabelled_batch)

                    for batch_inx in range(len(pred_dicts)):
                        self.frame_recall[unlabelled_batch['frame_id'][batch_inx]] = recall_dicts

                        self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                        all_frames.append(unlabelled_batch['frame_id'][batch_inx])

                        iou_scores = pred_dicts[batch_inx]['pred_scores']
                        select_dic[unlabelled_batch['frame_id'][batch_inx]] = torch.mean(iou_scores) # select

                if self.rank == 0:
                    pbar.update()
                    # pbar.set_postfix(disp_dict)
                    pbar.refresh()

            if self.rank == 0:
                pbar.close()
        if self.cfg.DATA_CONFIG.DATASET == "NuScenesDataset":
            all_frames = list(self.unlabelled_set.lidar_paths)

        select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        unlabelled_sample_num = len(select_dic.keys())
        selected_frames = list(select_dic.keys())[:self.cfg.ACTIVE_TRAIN.SELECT_NUMS]
        # selected_frames, _ = zip(*self.pairs[:self.cfg.ACTIVE_TRAIN.SELECT_NUMS])
        self.report_open_world_recall(self.frame_recall, selected_frames)

        return selected_frames