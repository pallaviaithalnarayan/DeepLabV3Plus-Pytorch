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

# source activate /path/to/your/conda/env  # Replace with your conda environment path

# Change to the directory where the script is located
cd ~/pallavi/DeepLabV3Plus-Pytorch/  # Replace with your project's directory

# Run the Python training script
python main.py                           # Run your training script

# Deactivate the Conda environment
conda deactivate

#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=any #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=60G          # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /work/svenka2s/pallavi/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /work/svenka2s/pallavi/job_tf.%N.%j.err  # filename for STDERR

# activate environment
source ~/miniconda3/bin/activate ~/miniconda3/envs/pallavi

# locate to your root directory
cd /work/svenka2s/pallavi/

python poc.py
