export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=(`echo $CUDA_VISIBLE_DEVICES | tr ',' ' '`)
num_gpus=${#gpus[@]}
port=2135${gpus[0]}
steps=400
cfg=2.0
# start time, sleep before it 
timestamp=$(date "+%Y%m%d%H%M%S")
run_name="llama_gen_xl_${timestamp}_steps_${steps}_384_c2i_8_vlm_rewards_cfg_${cfg}"

data_rt="<YOUR DATA ROOT>"

torchrun --nproc_per_node "$num_gpus" --nnodes "1" --master_addr "localhost" --master_port "$port" \
    img_gen_grpo_train.py \
    --model_name_or_path="custom/noneexists" \
    --dataset_name $data_rt/imagenet \
    --image_root $data_rt/imagenet \
    --run_name $run_name \
    --output_dir checkpoints/$run_name \
    --vq_ckpt './pretrained_models/vq_ds16_c2i.pt'\
    --gpt_ckpt './pretrained_models/c2i_XL_384.pt'\
    --gpt_model "GPT-XL"\
    --gpt_type c2i \
    --image_size 384 \
    --dropout_p 0.0 \
    --token_dropout_p 0.0 \
    --gen_cfg_cfg_scale $cfg \
    --num_generations 8 \
    --top_p 1.0 \
    --top_k 0 \
    --data_seed 42 \
    --reward_funcs clip_distance clip_quantize hpsv2 hpsv2_quantize maniqa qwenvl_3b qwenvl_fake qwenvl_weird \
    --reward_weights 1 1 1 1 1 1 1 1 \
    --beta 0.1 \
    --num_iterations 1 \
    --max_prompt_length 4096 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --logging_step 1 \
    --bf16 true \
    --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 20 \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 6 \
    --exit_step  ${steps}\
    --report_to tensorboard
