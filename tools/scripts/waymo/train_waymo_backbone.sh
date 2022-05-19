#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=wa_backbone
#SBATCH --time=400:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/waymo/Training_waymo_backbone_out_select_400.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/waymo/Training_waymo_backbone_error_select_400.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

module load anaconda/3.5
module load git-2.19.1-gcc-7.3.0-swjt5hp
source activate /scratch/itee/user/envs/active_3D

# tcp_port = 18880
srun python train.py --cfg_file cfgs/active-waymo_models/pv_rcnn_active_random.yaml --workers 3 --batch_size 6 --fix_random_seed --max_ckpt_save_num 200
# srun python -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/active-waymo_models/pv_rcnn_active_random.yaml --workers 3 --batch_size 20 --fix_random_seed --max_ckpt_save_num 200 --tcp_port $((tcp_port + 0)) 
# srun python -m torch.distributed.launch --nproc_per_node=4 test.py --launcher pytorch --cfg_file cfgs/active-waymo_models/pv_rcnn_active_random.yaml --batch_size 48 --eval_all --tcp_port $((tcp_port + 1)) && \
# srun python -m torch.distributed.launch --nproc_per_node=4 visualize.py --launcher pytorch --nproc_per_node=3 --cfg_file cfgs/active-kitti_models/pv_rcnn_active_random.yaml --batch_size 48 --eval_all --tcp_port $((tcp_port + 2))
# srun python train.py --cfg_file cfgs/active-waymo_models/pv_rcnn_active_random_1.yaml --batch_size 6 --fix_random_seed --max_ckpt_save_num 200 && \
# srun python train.py --cfg_file cfgs/active-waymo_models/pv_rcnn_active_random_2.yaml --batch_size 6 --fix_random_seed --max_ckpt_save_num 200

