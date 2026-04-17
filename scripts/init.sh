#!/usr/bin/env bash
#SBATCH -J init
#SBATCH -p vip_gpu_a800_scwb054
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=logs/init.out
#SBATCH --error=logs/init.err


source ~/.bashrc
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
fi


CONDA_BASE=/data/home/scwb054/run/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate osla

module load cuda/12.8
module load cudnn/9.11.0.98_cuda12

cd /data/home/scwb054/run/H800/finetune/open

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/home/scwb054/run/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL="1"
export TMPDIR=/data/home/scwb054/run/H800/SOLAR/temp

##########################################################

mkdir -p ./llama/svd_init_models
mkdir -p ./llama/logs/init
mkdir -p ./llama/logs/init/svd

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
NCCL_ASYNC_ERROR_HANDLING=1 \
SVD_LOG_ENABLE=1 \
SVD_LOG_STDOUT=0 \
SVD_LOG_FILE=./llama/logs/init/svd/svd_init.jsonl \
PYTHONUNBUFFERED=1 \
python init.py \
  --model_path /data/home/scwb054/run/MODEL/Llama-2-7b-hf \
  --save_path ./llama/svd_init_models \
  --task_name math \
  --hf_dataset meta-math/MetaMathQA  \
  --hf_split 'train[:256]' \
  --svd_rank 128 \
  --lora_alpha 128 \
  --init_method grad-kfac \
  --amp_dtype bf16 \
  --max_len 512 \
  --layers_split 1 \
  --layer_split_start 6 \
  2>&1 | tee ./llama/logs/init/init-code-grad-kfac-r64.log


#####################################################
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
NCCL_ASYNC_ERROR_HANDLING=1 \
SVD_LOG_ENABLE=1 \
SVD_LOG_STDOUT=0 \
SVD_LOG_FILE=./llama/logs/init/svd/svd_init.jsonl \
PYTHONUNBUFFERED=1 \
python init.py \
  --model_path /data/home/scwb054/run/MODEL/Llama-2-7b-hf \
  --save_path ./llama/svd_init_models \
  --task_name code \
  --hf_dataset m-a-p/CodeFeedback-Filtered-Instruction  \
  --hf_split 'train[:256]' \
  --svd_rank 128 \
  --lora_alpha 128 \
  --init_method grad-kfac \
  --amp_dtype bf16 \
  --max_len 512 \
  --layers_split 1 \
  --layer_split_start 4 \
  2>&1 | tee ./llama/logs/init/init-code-grad-kfac-r64.log


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
NCCL_ASYNC_ERROR_HANDLING=1 \
SVD_LOG_ENABLE=1 \
SVD_LOG_STDOUT=0 \
SVD_LOG_FILE=./llama/logs/init/svd/svd_init.jsonl \
PYTHONUNBUFFERED=1 \
python init.py \
  --model_path /data/home/scwb054/run/MODEL/Llama-2-7b-hf \
  --save_path ./llama/svd_init_models \
  --task_name instruction \
  --hf_dataset YeungNLP/WizardLM_evol_instruct_V2_143k  \
  --hf_split 'train[:256]' \
  --svd_rank 128 \
  --lora_alpha 128 \
  --init_method grad-kfac \
  --amp_dtype bf16 \
  --max_len 512 \
  --layers_split 1 \
  --layer_split_start 4 \
  2>&1 | tee ./llama/logs/init/init-instruction-grad-kfac-r64.log