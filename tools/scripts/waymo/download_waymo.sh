#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Download_waymo
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/Download_waymo_out.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/Download_waymo_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

module load anaconda/3.5
module load git-2.19.1-gcc-7.3.0-swjt5hp
source activate /scratch/itee/user/envs/active_3D

gsutil -m cp -r \
  "gs://waymo_open_dataset_v_1_2_0/testing" \
  "gs://waymo_open_dataset_v_1_2_0/training" \
  "gs://waymo_open_dataset_v_1_2_0/validation" \
  /clusterdata/user/data/DA_data/waymo/.