# python cli.py --model_name "bert-base-uncased" --subset_name "mrpc" --tokenizer_name "bert-base-uncased" --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "f1"
# python cli.py --model_name "bert-base-uncased" --subset_name "cola" --tokenizer_name "bert-base-uncased" --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "matthews_correlation"
# python cli.py --model_name "bert-base-uncased" --subset_name "rte"  --tokenizer_name "bert-base-uncased" --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "accuracy"
# python cli.py --model_name "bert-base-uncased" --subset_name "sst2" --tokenizer_name "bert-base-uncased" --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "accuracy"
python cli.py --model_name "bert-base-uncased" --subset_name "qnli" --tokenizer_name "bert-base-uncased" --tokenizer_truncation "longest_first" --tokenizer_max_length 256 --use_wandb recursive --target_metrics "accuracy"