#!/bin/bash
#SBATCH -J java
#SBATCH --gres=gpu:2
#SBATCH -p cas_v100_2
#SBATCH -N 1
#SBATCH -e %j.stderr
#SBATCH -o %j.stdout
#SBATCH --time=1:00:00
#SBATCH --comment=etc
module purge
module load gcc/8.3.0 cmake/3.12.3 cuda/10.1 python/3.7.1
~/.conda/envs/torch7/bin/python main.py --itr_before_train 33 --action_space 0 --alpha 1.0 --beta 0.0 --aux_reward 0.0 --scale_reward 0.01 --lr_scale 0.0 --out_activation 1.0 --init_noise_scale 0.05 --file_prefix $SLURM_JOBID

