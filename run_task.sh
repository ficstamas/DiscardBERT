while getopts g:m flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

tasks=("qqp" "stsb" "mnli_matched" "mnli_mismatched")
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
    python cli.py --wandb_project "recursive_nldb" --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda:$gpu" --model_name $model --subset_name "$task" --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "$metric" --recursive_steps "off_diagonal"
  done
done