#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Train_kitti
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/Training_kitti_random_out.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/Training_kitti_random_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

module load anaconda/3.5
module load git-2.19.1-gcc-7.3.0-swjt5hp
source activate /scratch/itee/user/envs/active_3D

srun python train.py --cfg_file cfgs/active-kitti_models/pv_rcnn_active_random.yaml --batch_size 6 --fix_random_seed --max_ckpt_save_num 200 
