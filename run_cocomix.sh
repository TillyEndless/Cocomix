#!/bin/bash

# 初始化 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cocomix

# 设置 WandB（需先登录或写入 .netrc）
export WANDB_API_KEY=ba89fea12f0ffd5e6cf6218effe78032f523f5b7
export WANDB_MODE=online

# 避免内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 代理设置（如使用）
export https_proxy=http://127.0.0.1:20171
export http_proxy=http://127.0.0.1:20171
export all_proxy=socks5h://127.0.0.1:20170

# Huggingface 缓存目录
export HF_HOME=/data/hf_cache
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

# 使用混合精度加速并节省显存
export ACCELERATE_MIXED_PRECISION=fp16

# 设置全部 GPU
export CUDA_VISIBLE_DEVICES=2

# 保存路径（需确保与训练配置中的 log_path 一致）
SAVE_DIR="./logs/openwebtext/openai-community/gpt2_embd512_L8_H8/cocomix_bs256_ctx512_lam0.1_seed_22"
CKPT_PATH="$SAVE_DIR/step_36000"

# 自动恢复 checkpoint（如果存在）
if [ -d "$CKPT_PATH" ]; then
	echo "Resuming from checkpoint: $CKPT_PATH"
	LOAD_PATH="++load_path=$CKPT_PATH"
else
	echo "No checkpoint found, training from scratch."
	LOAD_PATH=""
fi

# 启动训练
accelerate launch \
	 --config_file ./conf/fsdp_bf16.yaml \
	 --num_processes=1 \
	 main.py \
	     setup=gpt2_69m_cocomix \
	         hydra.run.dir=$SAVE_DIR \
		     ++concept_dim=32768 \
		         ++concept_num=32 \
			     ++block_size=512 \
			         ++update_batch_size=8 \
				     ++grad_acc_steps=32 \
				         ++batch_size_eval=8 \
					     ++attn_implementation=eager \
					         ++torch_dtype=float16 \
						     $LOAD_PATH
