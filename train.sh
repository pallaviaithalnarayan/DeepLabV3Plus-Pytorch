#!/bin/bash
# ==============================================================================
# SCRIPT:        train.sh
# DESCRIPTION:   Submits a job for model training in a specified environment.
# AUTHOR:        Pallavi Aithal
# EMAIL:         pallavi.narayan@smail.inf.h-brs.de
# DATE:          2024-11-14
# VERSION:       1.0.0
# ==============================================================================

#SBATCH --job-name=train_job            # Job name
#SBATCH --output /pallavi/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /pallavi/job_tf.%N.%j.err  # filename for STDERR
#SBATCH --time=72:00:00                 # Time limit in HH:MM:SS
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --mem=32GB                      # Memory per node (adjust as needed)
#SBATCH --gres=gpu:1                    # Number of GPUs (adjust if needed)
#SBATCH --ntasks-per-node=64    # number of cores

# Load the necessary modules (e.g., anaconda, CUDA)
# module load cuda/11.7
source ~/miniconda3/bin/activate ~/miniconda3/envs/pallavi

# Job parameters
JOB_NAME="train"
PYTHON_SCRIPT="main.py"
CONDA_ENV="pallavi"  # Name of your conda environment
JOB_ARGS="--model deeplabv3plus_mobilenet --num_classes 7 --dataset customdata --gpu_id 0 --lr 0.001 --crop_size 512 --batch_size 16 --output_stride 16 --data_root ./datasets/citypark/ --save_val_results --separable_conv --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"

# source ~/3/etc/profile.d/conda.sh
# conda activate $CONDA_ENV
# source activate /path/to/your/conda/env  # Replace with your conda environment path

# Change to the directory where the script is located
cd ~/pallavi/DeepLabV3Plus-Pytorch/  # Replace with your project's directory

# Run the Python training script
python $PYTHON_SCRIPT $JOB_ARGS                        # Run your training script

# # Deactivate the Conda environment
# conda deactivate
