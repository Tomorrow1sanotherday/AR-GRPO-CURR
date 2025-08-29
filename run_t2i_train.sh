export CUDA_VISIBLE_DEVICES=4,5,6,7
gpus=(`echo $CUDA_VISIBLE_DEVICES | tr ',' ' '`)
num_gpus=${#gpus[@]}
port=2135${gpus[0]}
steps=600
cfg=4.0

# start time, sleep before it 
timestamp=$(date "+%Y%m%d%H%M%S")
run_name="llama_gen_${timestamp}_steps_${steps}_t2i_8_vlm_rewards_base_4_vlm_rewards_d4_coco_cfg_${cfg}"

torchrun --nproc_per_node "$num_gpus" --nnodes "1" --master_addr "localhost" --master_port "$port" \
    img_gen_grpo_train.py \
    --model_name_or_path="custom/noneexists" \
    --dataset_name ./dataset/coco_captions_30000.json \
    --image_root $data_rt/coco \
    --run_name $run_name \
    --output_dir checkpoints/$run_name \
    --vq_ckpt './pretrained_models/vq_ds16_t2i.pt'\
    --gpt_ckpt './pretrained_models/t2i_XL_stage1_256.pt'\
    --gpt_type t2i \
    --gpt_model "GPT-XL" \
    --cls_token_num 120 \
    --image_size 256 \
    --dropout_p 0.0 \
    --token_dropout_p 0.0 \
    --gen_cfg_cfg_scale ${cfg} \
    --num_generations 8 \
    --top_p 1.0 \
    --top_k 0 \
    --data_seed 42 \
    --reward_funcs vqa_score  \
    --reward_weights 1\
    --beta 1.0 \
    --num_iterations 1 \
    --max_prompt_length 4096 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --logging_step 1 \
    --bf16 true \
    --torch_dtype bfloat16 \
    --force_model_bf16 True \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 20 \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 6 \
    --exit_step  ${steps}\
    --report_to tensorboard \
    --sample_strategy timestep