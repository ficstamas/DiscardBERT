while getopts ":g:m:" flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

tasks=("rte" "mrpc" "cola" "sst2" "qnli" "stsb")
seeds=(0 1 2 3 4)


for task in ${tasks[@]}
do
  metric="accuracy"
  if [[ "$task" == "qqp" ]]; then
    metric="f1"
  elif [[ "$task" == "stsb" ]]; then
    metric="spearmanr"
  fi

  for seed in ${seeds[@]}
  do
    python cli.py --wandb_project "recursive_nldb_small_models" --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda" --device_id "$gpu" --model_name $model --subset_name "$task" --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "$metric" --recursive_steps "off_diagonal"
  done
done


tasks=("pos" "ner")

for task in ${tasks[@]}
do
  metric="accuracy"
  if [[ "$task" == "ner" ]]; then
    metric="f1"
  fi

  for seed in ${seeds[@]}
  do
    python cli.py --wandb_project "recursive_nldb_small_models" --model_type "token" --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda" --device_id "$gpu" --dataset_name "conll" --model_name $model --subset_name "$task" --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "$metric" --recursive_steps "off_diagonal"
  done
done