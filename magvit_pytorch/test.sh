#!/bin/bash
#SBATCH --output=output_report-%j.out
#SBATCH --job-name=llama2
#SBATCH --gres=gpu:A6000:4
# SBATCH --partition=short-inst
#SBATCH --time=1-1:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=wenkail@cs.cmu.edu

source ~/.bashrc
conda activate mc
cd ~/VQGAN-pytorch
python3 test.py