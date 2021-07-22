#!/bin/bash
#SBATCH --job-name=hello-world-job
#SBATCH --ntasks=30
#SBATCH --output=%x_%j.out

mpirun python3 /home/ubuntu/hello.py
