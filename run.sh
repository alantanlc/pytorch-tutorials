#!/bin/sh

#SBATCH -o gpu-job-%j.output
#SBATCH -p PV1003q
#SBATCH --gres=gpu:1
#SBATCH -n 1

module load cuda90/toolkit

python $1

echo Completed
