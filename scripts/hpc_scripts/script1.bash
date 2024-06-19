#!/bin/bash -l
#SBATCH --mail-user=boghos.youseef.001@student.uni.lu
#SBATCH --mail-type=ALL
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --time=01:30:00
#SBATCH -p gpu
#SBATCH --mem=10GB

python3 .\src\main.py train -s 512 512 -e 100 -n patch_model