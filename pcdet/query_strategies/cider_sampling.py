
import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
import torch.nn.functional as F
import tqdm
import faiss
class CiderSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(CiderSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

            


    def query(self, leave_pbar=True, cur_epoch=None):
        select_dic = {}

        
        self.model.eval()
        unlabeled_embeddings = []
        labeled_embeddings = []
        unlabeled_frame_ids = []
        frame_embeddings = {}
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
                # https://github.com/Jingkang50/OpenOOD/blob/e1b01af0cb4fe293df12fabd05840e130d96c30b/openood/postprocessors/cider_postprocessor.py
                for batch_inx in range(len(pred_dicts)):
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    unlabeled_frame_ids.append(unlabelled_batch['frame_id'][batch_inx])
                    self.frame_recall[unlabelled_batch['frame_id'][batch_inx]] = _

                    embeddings = pred_dicts[batch_inx]['embeddings'].view(-1, 128*self.cfg.MODEL.ROI_HEAD.SHARED_FC[-1])
                    frame_embeddings[unlabelled_batch['frame_id'][batch_inx]] = embeddings

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        activation_log =torch.concat(list(frame_embeddings.values()), dim=0).cpu().numpy()
        feature = list(frame_embeddings.values())[0]
        index = faiss.IndexFlatL2(feature.shape[1])
        index.add(activation_log)

        frame_dist = {}
        for k in frame_embeddings.keys():
            feature = frame_embeddings[k]
            D, _ = index.search(
                feature.cpu().numpy(),
                # feature is already normalized within net
                5,
            )
            kth_dist = -D[:, -1]
            frame_dist[k] = kth_dist

        # sort and get selected_frames
        frame_dist = dict(sorted(frame_dist.items(), key=lambda item: item[1]))
        selected_frames = list(frame_dist.keys())[:int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS)]
        self.report_open_world_recall(self.frame_recall, selected_frames)

        return selected_frames
