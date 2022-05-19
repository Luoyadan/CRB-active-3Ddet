#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=Unzip_waymo
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -o /clusterdata/user/data/active-3D-detection/logs/Unzip_waymo_out.txt
#SBATCH -e /clusterdata/user/data/active-3D-detection/logs/Unzip_waymo_error.txt
#SBATCH --partition=gpu

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20


for f in /scratch/itee/user/DA_data/waymo/training/*.tar
do
  tar xvf "$f" -C /scratch/itee/user/DA_data/waymo/training
done

for f in /scratch/itee/user/DA_data/waymo/validation/*.tar
do
  tar xvf "$f" -C /scratch/itee/user/DA_data/waymo/validation
done


for f in /scratch/itee/user/DA_data/waymo/testing/*.tar
do
  tar xvf "$f" -C /scratch/itee/user/DA_data/waymo/testing
done