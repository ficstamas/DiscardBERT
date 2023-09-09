seeds=(0 1 2 3 4)
models=("bert-large-uncased")

for model in ${models[@]}
do
for seed in ${seeds[@]}
do
python cli.py --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda" --model_name $model --subset_name "cola" --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "matthews_correlation" --recursive_steps "off_diagonal"
python cli.py --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda" --model_name $model --subset_name "mrpc" --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "f1" --recursive_steps "off_diagonal"
python cli.py --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda" --model_name $model --subset_name "rte"  --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "accuracy" --recursive_steps "off_diagonal"
done
done
