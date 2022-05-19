#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=wa_crb
#SBATCH --time=400:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/waymo/Training_waymo_crb_select400_out.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/waymo/Training_waymo_crb_select400_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

module load anaconda/3.5
module load git-2.19.1-gcc-7.3.0-swjt5hp
source activate /scratch/itee/user/envs/active_3D


srun python train.py --cfg_file cfgs/active-waymo_models/pv_rcnn_active_crb.yaml --workers 3 --batch_size 6 --fix_random_seed --max_ckpt_save_num 200 

source activate /scratch/itee/user/envs/active_3D_waymo_test

srun python test.py --cfg_file cfgs/active-waymo_models/pv_rcnn_active_crb.yaml --workers 3 --batch_size 16 --eval_all && \
srun python visualize.py --cfg_file cfgs/active-waymo_models/pv_rcnn_active_crb.yaml --batch_size 16 --eval_all 


