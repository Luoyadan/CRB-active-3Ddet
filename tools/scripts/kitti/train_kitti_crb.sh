#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=crb
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/kitti/Training_kitti_crb_out.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/kitti/Training_kitti_crb_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

module load anaconda/3.5
module load git-2.19.1-gcc-7.3.0-swjt5hp
source activate /scratch/itee/user/envs/active_3D

srun python train.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb.yaml --batch_size 6 --fix_random_seed --max_ckpt_save_num 200 && \
srun python test.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb.yaml --batch_size 16 --eval_all && \
srun python visualize.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb.yaml --batch_size 16 --eval_all 

srun python train.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb_1.yaml --batch_size 6 --fix_random_seed --max_ckpt_save_num 200 
srun python test.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb_1.yaml --batch_size 16 --eval_all && \
srun python visualize.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb_1.yaml --batch_size 16 --eval_all

srun python train.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb_2.yaml --batch_size 6 --fix_random_seed --max_ckpt_save_num 200 
srun python test.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb_2.yaml --batch_size 16 --eval_all && \
srun python visualize.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_crb_2.yaml --batch_size 16 --eval_all 