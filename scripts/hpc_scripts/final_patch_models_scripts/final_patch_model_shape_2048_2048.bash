#!/bin/bash -l
#SBATCH --mail-user=boghos.youseef.001@student.uni.lu
#SBATCH --mail-type=ALL
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --qos normal
#SBATCH --time=2:00:00
#SBATCH -p gpu
#SBATCH --mem=10GB


python3 main.py train -mt p -s 2048 2048 -e 100 -n final_patch_model -ds 0 -v

