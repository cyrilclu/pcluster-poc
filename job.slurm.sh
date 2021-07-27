#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -o uc2_%j.out

srun /home/ubuntu/uc2_ddp.sh
