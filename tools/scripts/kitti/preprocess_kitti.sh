#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Preprocess_kitti
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/Preprocess_kitti_out.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/Preprocess_kitti_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

module load anaconda/3.5
module load git-2.19.1-gcc-7.3.0-swjt5hp
source activate /scratch/itee/user/envs/active_3D

srun python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos cfgs/dataset_configs/kitti_dataset.yaml