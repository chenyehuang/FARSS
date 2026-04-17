#!/usr/bin/env bash
#SBATCH -J train
#SBATCH -p vip_gpu_a800_scwb054
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err

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


export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTHONUNBUFFERED=1 deepspeed \
  --master_addr 127.0.0.1 \
  --master_port 29501 \
  --include localhost:0,1,2,3,4,5,6,7 \
  train.py \
  --deepspeed configs/stage2.conf \
  --model_name_or_path ./llama/svd_init_models/math-grad-kfac-r128 \
  --adapter_name_or_path ./llama/svd_init_models/math-grad-kfac-r128/lora \
  --method_type grad-kfac \
  --output_dir ./llama/output/math/math-grad-kfac-N100k-LR2e-5-E1-r128 \
  --data_path meta-math/MetaMathQA \
  --dataset_split 'train[:100000]' \
  --dataset_field query response \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --model_max_length 512 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --warmup_steps 10 \
  --lr_scheduler_type linear \
  --save_strategy epoch \
  --bf16 True \
  --logging_strategy steps \
  --logging_steps 10 \
  --logging_first_step True \
  --max_grad_norm 1.0 \
  --report_to tensorboard \
  > ./llama/logs/train/math-grad-kfac-N100k-LR2e-5-E1-r128.log 2>&1


############################################

PYTHONUNBUFFERED=1 deepspeed \
  --master_addr 127.0.0.1 \
  --master_port 29501 \
  --include localhost:0,1,2,3,4,5,6,7 \
  train.py \
  --deepspeed configs/stage2.conf \
  --model_name_or_path ./llama/svd_init_models/code-grad-kfac-r128 \
  --adapter_name_or_path ./llama/svd_init_models/code-grad-kfac-r128/lora \
  --method_type grad-kfac \
  --output_dir ./llama/output/code/code-grad-kfac-N100k-LR2e-5-E1-r128 \
  --data_path m-a-p/CodeFeedback-Filtered-Instruction \
  --dataset_split 'train[:100000]' \
  --dataset_field query answer \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --model_max_length 512 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --warmup_steps 10 \
  --lr_scheduler_type linear \
  --save_strategy epoch \
  --bf16 True \
  --logging_strategy steps \
  --logging_steps 10 \
  --logging_first_step True \
  --max_grad_norm 1.0 \
  --report_to tensorboard \
  > ./llama/logs/train/code-grad-kfac-N100k-LR2e-5-E1-r128.log 2>&1

#####################################################

PYTHONUNBUFFERED=1 deepspeed \
  --master_addr 127.0.0.1 \
  --master_port 29501 \
  --include localhost:0,1,2,3,4,5,6,7 \
  train.py \
  --deepspeed configs/stage2.conf \
  --model_name_or_path ./llama/svd_init_models/instruction-grad-kfac-r128 \
  --adapter_name_or_path ./llama/svd_init_models/instruction-grad-kfac-r128/lora \
  --method_type grad-kfac \
  --output_dir ./llama/output/instruction/instruction-grad-kfac-N100k-LR2e-5-E1-r128 \
  --data_path YeungNLP/WizardLM_evol_instruct_V2_143k \
  --dataset_split 'train[:100000]' \
  --dataset_field instruction response \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --model_max_length 512 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --warmup_steps 10 \
  --lr_scheduler_type linear \
  --save_strategy epoch \
  --bf16 True \
  --logging_strategy steps \
  --logging_steps 10 \
  --logging_first_step True \
  --max_grad_norm 1.0 \
  --report_to tensorboard \
  > ./llama/logs/train/instruction-grad-kfac-N100k-LR2e-5-E1-r128.log 2>&1