tasks=("rte" "cola" "mrpc" "sst2" "qnli")
inits=("pdf" "pfdf")


for task in ${tasks[@]}
do
for init in ${inits[@]}
do
for i in `seq 0 2 4`
do
for j in `seq ${i+2} 2 4`
do
	python cli.py --model_name "distilbert-base-uncased" --tokenizer_name "distilbert-base-uncased" --subset_name $task --initial_model $init --use_wandb --wandb_project "lrec" --range $i $j --elimination "range" --device "cuda"
done
done
done
done
