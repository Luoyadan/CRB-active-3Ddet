
import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
import torch.nn.functional as F
import tqdm

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    i = 0
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            i += 1
            m.train()
    print('**found {} Dropout layers for random sampling'.format(i))

class MonteCarloSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(MonteCarloSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        enable_dropout(self.model)
     

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
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    
                    # did not apply batch mask -> directly output 
                    cls_var = torch.var(torch.sigmoid(pred_dicts[batch_inx]['rcnn_cls_raw']), 0).mean()
                    # reg_var = torch.var(pred_dicts[0]['rcnn_reg'], 0).mean()
                    

                    """ Aggregate all the boxes values in one point cloud """
                    if self.cfg.ACTIVE_TRAIN.AGGREGATION == 'mean':
                        aggregated_values = cls_var
                    else:
                        raise NotImplementedError


                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = aggregated_values

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        # sort and get selected_frames
        select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        unlabelled_sample_num = len(select_dic.keys())


        selected_frames = list(select_dic.keys())[unlabelled_sample_num - self.cfg.ACTIVE_TRAIN.SELECT_NUMS:]
        self.report_open_world_recall(self.frame_recall, selected_frames)
        return selected_frames
