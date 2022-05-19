#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Preprocess_waymo
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/Preprocess_waymo_out.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/Preprocess_waymo_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

module load anaconda/3.5
module load git-2.19.1-gcc-7.3.0-swjt5hp
module load cuda/11.3.0
source activate /scratch/itee/user/envs/active_3D

srun python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file cfgs/dataset_configs/waymo_dataset.yaml