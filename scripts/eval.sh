#!/usr/bin/env bash
#SBATCH -J eval
#SBATCH -p vip_gpu_a800_scwb054
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err


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

cd /data/home/scwb054/run/H800/finetune/open

# sleep 1h
################################# math ################################
mkdir -p ./llama/logs/eval
base_model=./llama/svd_init_models/math-grad-kfac-r128
adapter=./llama/output/math/math-grad-kfac-N100k-LR2e-5-E1-r128/ft

log_file=./llama/logs/eval/math-grad-kfac-lr2e-5-r128.log

if [ ! -f $log_file ]; then
    touch $log_file
fi

export CUDA_VISIBLE_DEVICES=0
lm_eval --model hf \
    --model_args pretrained=${base_model},peft=${adapter} \
    --tasks triviaqa,nq_open,webqs \
    --device cuda:0 \
    --batch_size 64 \
    2>&1 | tee -a $log_file


adapter_path_root=./llama/output/math/math-grad-kfac-N100k-LR2e-5-E1-r128/
adapter_path=${adapter_path_root}/ft
output_path=${adapter_path_root}/merged

rm -rf ${output_path}
CUDA_VISIBLE_DEVICES=0 python merge_adapter_to_base_model.py \
    --base_model "${base_model}" \
    --adapter "${adapter_path}" \
    --output_path "${output_path}"

CUDA_VISIBLE_DEVICES=0 python inference/gsm8k_inference.py --model $output_path 2>&1 | tee -a $log_file
CUDA_VISIBLE_DEVICES=0 python inference/MATH_inference.py --model $output_path 2>&1 | tee -a $log_file

#################### code ##########################

mkdir -p ./llama/logs/eval
base_model=./llama/svd_init_models/code-grad-kfac-r128
adapter_path_root=./llama/output/code/code-grad-kfac-N100k-LR2e-5-E1-r128
adapter_path=${adapter_path_root}/ft
output_path=${adapter_path_root}/merged

rm -rf ${output_path}
CUDA_VISIBLE_DEVICES=0 python merge_adapter_to_base_model.py \
    --base_model "${base_model}" \
    --adapter "${adapter_path}" \
    --output_path "${output_path}"

log_file=./llama/logs/code_eval/code-grad-kfac-lr2e-5-r128.log

if [ ! -f $log_file ]; then
    touch $log_file
fi

export CUDA_VISIBLE_DEVICES=0
lm_eval --model hf \
    --model_args pretrained=${base_model},peft=${adapter_path} \
    --tasks triviaqa,nq_open,webqs \
    --device cuda:0 \
    --batch_size 64 \
    2>&1 | tee -a $log_file



export CUDA_VISIBLE_DEVICES=0
mkdir -p ./llama/logs/code_eval/farss
rm -rf ./llama/logs/code_eval/farss/*
python inference/gen_vllm.py --model $output_path --sub_task python --output_file ./llama/logs/code_eval/farss/python_response.jsonl 2>&1 | tee -a $log_file
python inference/code_process.py --path ./llama/logs/code_eval/farss/python_response.jsonl 2>&1 | tee -a $log_file
evalplus.evaluate --dataset humaneval --samples ./llama/logs/code_eval/farss/humaneval.jsonl 2>&1 | tee -a $log_file
evalplus.evaluate --dataset mbpp --samples ./llama/logs/code_eval/farss/mbpp.jsonl 2>&1 | tee -a $log_file
rm -rf ${output_path}


# ##################### instruction ##########################
mkdir -p ./llama/logs/eval

base_model=./llama/svd_init_models/instruction-grad-kfac-r128
adapter=./llama/output/instruction/instruction-grad-kfac-N100k-LR2e-5-E1-r128/ft

log_file=./llama/logs/eval/instruction-farss-lr2e-5-r128.log

if [ ! -f $log_file ]; then
    touch $log_file
fi

export CUDA_VISIBLE_DEVICES=0
lm_eval --model hf \
    --model_args pretrained=${base_model},peft=${adapter} \
    --tasks triviaqa,nq_open,webqs,ifeval \
    --device cuda:0 \
    --batch_size 64 \
    2>&1 | tee -a $log_file