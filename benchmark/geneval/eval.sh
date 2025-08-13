image_dir=$1
gpu=$2
test_name=`basename $image_dir`
out_file="test_results/res_${test_name}.jsonl"
export CUDA_VISIBLE_DEVICES=$gpu
#echo "geneval $image_dir"
python evaluation/evaluate_images.py \
"$image_dir" \
 --outfile  $out_file\
 --model-path "models"

 python evaluation/summary_scores.py $out_file