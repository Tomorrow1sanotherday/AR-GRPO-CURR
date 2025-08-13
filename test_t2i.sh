exp_name=$1
gpu=$2
step=$3
cfg=$4

if [[ $cfg == "" ]]
then
    cfg=2.0
fi

echo "eval ${exp_name} step ${step} on GPU ${gpu} with cfg ${cfg}"

if [[ $gpu == "" ]]
then
    gpu=0
fi

if [[ $step == "" ]]
then
    step=300
fi
out_rt=`realpath "test_results/vis_out_${exp_name}_${step}_${cfg}"`

export CUDA_VISIBLE_DEVICES=$gpu
python test_t2i.py --exp_name "$exp_name" --step "$step" --out_rt "$out_rt" --cfg $cfg
