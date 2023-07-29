#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1 --constraint=48G
#SBATCH --mem=64G
#SBATCH --job-name="RLHF-PPO-Sentiment"

source /iris/u/rafailov/.bashrc
conda activate trl
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sailhome/rafailov/.mujoco/mujoco210/bin:/usr/lib/nvidia
export WANDB_DIR=/iris/u/rafailov/LLMRL/CACHE/wandb/Detection/dir
export WANDB_CACHE_DIR=/iris/u/rafailov/LLMRL/CACHE/wandb/Detection/cache
export WANDB_DATA_DIR=/iris/u/rafailov/LLMRL/CACHE/wandb/Detection/data
export TRANSFORMERS_CACHE=/iris/u/rafailov/LLMRL/CACHE/hub
export HUGGINGFACE_HUB_CACHE=/iris/u/rafailov/LLMRL/CACHE/hub
export HF_HOME=/iris/u/rafailov/LLMRL/CACHE/hub
export HF_DATASETS_CACHE=/iris/u/rafailov/LLMRL/CACHE/datasets
export HF_MODULES_CACHE=/iris/u/rafailov/LLMRL/CACHE/modules
cd /iris/u/rafailov/transformers/trl/examples/scripts
which python
#echo $SLURM_JOB_GPU
#export GPUS=$SLURM_JOB_GPU

python3 -u sentiment_tuning.py

