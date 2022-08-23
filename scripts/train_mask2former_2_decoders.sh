#!/bin/bash
# 
#
# 
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=TrainMask2Former-Swin-B-2-DecLayer-Res5Only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=avg
#SBATCH --account=avg
#SBATCH --qos=avg
#SBATCH --gres=gpu:2
#SBATCH --constraint="rtx_a6000"
#SBATCH --mem-per-cpu=6G
#SBATCH --time=168:00:00
#SBATCH --output=logs/logs-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nnayal17@ku.edu.tr

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

module load anaconda/3.6
module load cuda/11.1
module load cudnn/8.1.1/cuda-11.X
module load gcc/9.3.0
module load nccl/2.9.6-1/cuda-11.0
source activate maskformer_env

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Example Job...!"
echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."
# Put Python script commands below

export DETECTRON2_DATASETS=/scratch/users/nnayal21/datasets
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export OPENBLAS_NUM_THREADS=22
export OMP_NUM_THREADS=22
export MKL_NUM_THREADS=22
export VECLIB_MAXIMUM_THREADS=22
export NUMEXPR_NUM_THREADS=22

python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k_variant.yaml \
  --num-gpus 2 --resume MODEL.WEIGHTS ./model_logs/mask2former_dec_layers_2_res5_only/model_0009999.pth SOLVER.IMS_PER_BATCH 16 SOLVER.BASE_LR 0.0001 DATALOADER.NUM_WORKERS 10 OUTPUT_DIR ./model_logs/mask2former_dec_layers_2_res5_only/ 

