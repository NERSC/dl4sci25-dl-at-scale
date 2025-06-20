#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -A trn011
#SBATCH -q regular
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=01:00:00
#SBATCH --image=nersc/pytorch:24.08.01
#SBATCH --reservation=dl4sci
#SBATCH --module=gpu,nccl-plugin
#SBATCH -J vit-era5-datapar
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/s/shas1693/data/dl-at-scale-training-data
LOGDIR=${SCRATCH}/dl-at-scale-training/logs
mkdir -p ${LOGDIR}
args="${@}"

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    python train.py ${args}
    "
