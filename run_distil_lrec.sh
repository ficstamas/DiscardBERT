tasks=("rte" "cola" "mrpc" "sst2" "qnli")
inits=("pdf")


for task in ${tasks[@]}
do
for init in ${inits[@]}
do
for seed in `seq 0 1 4`
do
	python cli.py --model_name "distilbert-base-uncased" --tokenizer_name "distilbert-base-uncased" --subset_name $task --initial_model $init --use_wandb --wandb_project "lrec" --range "0" "0" --seed $seed --elimination "range" --device "cuda" --tokenizer_truncation "longest_first" --tokenizer_max_length 256  simple
done
done
done
