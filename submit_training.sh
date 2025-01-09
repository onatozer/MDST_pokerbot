#!/bin/bash

#Indicate the account to charge the job to
#SBATCH --account=mdatascience_team

#Indicate a name to give the job. This can be anything, it's just a way to be able to easily track different jobs
#SBATCH --job-name=train_pokerbot

#Indicate where to send information about the job
#SBATCH --mail-user=ozeronat@umich.edu

#Indicate how often you want info about the job. In this case, you will receive an email when the job has ended
#SBATCH --mail-type=END

#Indicate how many nodes to run the job on
#SBATCH --nodes=1

#Indicate how many tasks to run on each node
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=8              # Number of CPUs per task
#SBATCH --mem-per-cpu=8000m            # Memory per CPU (8GB)
#SBATCH --time=10:00:00                # Maximum run time

#SBATCH --gpus=1                       # Number of GPUs to allocate
#SBATCH --constraint=gpu_a100         #really not sure that this is a thing
#SBATCH --partition=spgpu              

#Get rid of any modules that might still be running
module purge

# Create virtual env
# python3 -m venv venv #Only needs to be ran once on device
source venv/bin/activate

# pip3 install-r requirements.txt #Same thing here

# Run the desired program
nvidia-smi
python3 train_model.py