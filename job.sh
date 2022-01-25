#!/bin/bash
#SBATCH -J job                   # Sensible name for the job
#SBATCH -N 2                     # Allocate 2 nodes for the job
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --gres=gpu:1
##SBATCH -c 20
#SBATCH -t 00:10:00             # Upper time limit for the job
#SBATCH -p GPUQ
module load fosscuda/2019a
module load TensorFlow/1.13.1-Python-3.7.2
time mpirun python3 testmpi.py
