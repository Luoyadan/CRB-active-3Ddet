import torch
# print(torch.__version__)
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
import random
from pcdet.config import cfg
__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler

def build_active_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                            logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0,
                            active_training=None):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    labelled_set = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=True,
        logger=logger,
    )

    unlabelled_set = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=False,
        logger=logger,
    )

    if active_training is not None:
        # after selecting samples,
        # build new datasets and dataloaders during active training loop
        if cfg.DATA_CONFIG.DATASET == 'WaymoDataset':
            labelled_set.frame_ids, labelled_set.infos = \
                active_training[0], active_training[1]
            unlabelled_set.frame_ids, unlabelled_set.infos = \
                active_training[2], active_training[3]

        else: # kitti cases
            labelled_set.sample_id_list, labelled_set.kitti_infos = \
                active_training[0], active_training[1]
            unlabelled_set.sample_id_list, unlabelled_set.kitti_infos = \
                active_training[2], active_training[3]



    else:
        # Build pre-train datasets and dataloaders before active training loop

        if cfg.DATA_CONFIG.DATASET == 'WaymoDataset':

            infos = dataset.infos
            random.shuffle(infos)
            labelled_set.infos = infos[:cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS]
            unlabelled_set.infos = infos[cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS:]

            for info in labelled_set.infos:
                labelled_set.frame_ids.append(info["frame_id"])
            for info in unlabelled_set.infos:
                unlabelled_set.frame_ids.append(info["frame_id"])
            
        else: # kitti case
            pairs = list(zip(dataset.sample_id_list, dataset.kitti_infos))
            random.shuffle(pairs)
            # labelled_set, unlabelled_set = copy.deepcopy(dataset), copy.deepcopy(dataset)
            labelled_set.sample_id_list, labelled_set.kitti_infos = \
                zip(*pairs[:cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS])
            unlabelled_set.sample_id_list, unlabelled_set.kitti_infos = \
                zip(*pairs[cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS:])


    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        labelled_set.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
        unlabelled_set.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler_labelled = torch.utils.data.distributed.DistributedSampler(labelled_set)
            sampler_unlabelled = torch.utils.data.distributed.DistributedSampler(unlabelled_set)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler_labelled = DistributedSampler(labelled_set, world_size, rank, shuffle=False)
            sampler_unlabelled = DistributedSampler(unlabelled_set, world_size, rank, shuffle=False)
    else:
        sampler_labelled, sampler_unlabelled =  None, None


    dataloader_labelled = DataLoader(
        labelled_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler_labelled is None) and training, collate_fn=labelled_set.collate_batch,
        drop_last=False, sampler=sampler_labelled, timeout=0
        )
    dataloader_unlabelled = DataLoader(
        unlabelled_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler_unlabelled is None) and training, collate_fn=unlabelled_set.collate_batch,
        drop_last=False, sampler=sampler_unlabelled, timeout=0
        )

    del dataset
    return labelled_set, unlabelled_set, \
           dataloader_labelled, dataloader_unlabelled, \
           sampler_labelled, sampler_unlabelled
