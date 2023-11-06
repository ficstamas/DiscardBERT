seeds=(0 1 2 3 4)
models=("distilbert-base-uncased")

for model in ${models[@]}
do
for seed in ${seeds[@]}
do
python cli.py --model_type "token" --wandb_project "recursive" --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda" --device_id 2 --model_name $model --dataset_name "conll" --subset_name "ner" --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "f1" --recursive_steps "off_diagonal"
python cli.py --model_type "token" --wandb_project "recursive" --seed $seed --batch_size 16 --initial_model "pfdf" --device "cuda" --device_id 2 --model_name $model --dataset_name "conll" --subset_name "pos" --tokenizer_name $model --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "f1" --recursive_steps "off_diagonal"
done
done
