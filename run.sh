#!/bin/sh

#SBATCH -o gpu-job-%j.output
#SBATCH -p K20q
#SBATCH --gres=gpu:1
#SBATCH -n 1

module load cuda90/toolkit

python $1

echo Completed
