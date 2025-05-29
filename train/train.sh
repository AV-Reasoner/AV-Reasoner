WANDB_PROJECT=rft
WANDB_NAME=av_reasoner


mkdir -p "../ckpt/$WANDB_PROJECT/$WANDB_NAME"

torchrun --nproc_per_node=8 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    grpo_uni.py \
    --output_dir "../ckpt/$WANDB_PROJECT/$WANDB_NAME" \
    --model_name_or_path "../ckpt/ola_7b" \
    --deepspeed "../scripts/zero2_offload.json" \
    --dataset_name xxx \
    --jsonl_path "../dataset/aba_counting_rft.jsonl" \
    --max_prompt_length 4096 \
    --max_completion_length 1024 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name "$WANDB_NAME" \
    --save_steps 20 \
    --save_only_model false \
    --save_total_limit 1 \
    --num_generations 8