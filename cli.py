import sys
from argparse import ArgumentParser
from discardbert.training_loop import Loop, ModelType, DatasetType, SubsetType, TrainingType, EliminationType
from typing import get_args
from discardbert.optimizer import STR2OPTIM, OptimType


args_ = ArgumentParser("DiscardBERT")

# general parameters
args_.add_argument("--model_name", type=str, default="prajjwal1/bert-medium")
args_.add_argument("--model_type", type=str, choices=get_args(ModelType), default="sequence")
args_.add_argument("--dataset_name", type=str, choices=get_args(DatasetType), default="glue")
args_.add_argument("--subset_name", type=str, choices=get_args(SubsetType), default="mrpc")
args_.add_argument("--pre_evaluation", action="store_true")

args_.add_argument("--learning_rate", type=float, default=2e-5)
args_.add_argument("--num_epoch", type=int, default=3)
args_.add_argument("--batch_size", type=int, default=16)

# optimizer parameters
args_.add_argument("--optimizer", type=str, choices=get_args(OptimType), default="adamw")
args_.add_argument("--optimizer_adamw_betas", action="extend", nargs=2, type=float, dest="elimination_range",
                   default=(0.9, 0.999))
args_.add_argument("--optimizer_adamw_eps", type=float, default=1e-8)
args_.add_argument("--optimizer_adamw_weight_decay", type=float, default=0.01)
args_.add_argument("--optimizer_sgd_momentum", type=float, default=0.01)
args_.add_argument("--optimizer_sgd_weight_decay", type=float, default=0)

# tokenizer parameters
args_.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-medium")
args_.add_argument("--tokenizer_truncation", type=str,
                   choices=['longest_first', 'only_first', 'only_second', 'do_not_truncate'], default='do_not_truncate')
args_.add_argument("--tokenizer_padding", type=str,
                   choices=['longest', 'max_length', 'do_not_pad'], default='do_not_pad')
args_.add_argument("--tokenizer_max_length", type=int)

# elimination parameters
args_.add_argument("--elimination", type=str, choices=get_args(EliminationType))
args_.add_argument("--range", action="extend", nargs=2, type=int, dest="elimination_range")
args_.add_argument("--exact_layers", action="extend", nargs="+", type=int, dest="elimination_exact_layers")

# lr scheduler
args_.add_argument("--lr_scheduler", type=str, default="linear")
args_.add_argument("--lr_scheduler_num_warmup_steps", type=int, default=0)
args_.add_argument("--lr_scheduler_num_training_steps", type=int)

# logging
args_.add_argument("--logging_interval", type=int, default=100)
args_.add_argument("--use_wandb", action="store_true")
args_.add_argument("--wandb_project", type=str, default="huggingface")
args_.add_argument("--wandb_entity", type=str, default="szegedai-semantics")

# subprograms : training procedures
subparser_training = args_.add_subparsers(help="Training procedure", dest="training_command")
simple = subparser_training.add_parser("simple", help="Simple/Normal training procedure")
recursive = subparser_training.add_parser("recursive", help="Recursive training procedure")

args, _ = args_.parse_known_args()

lr = args.learning_rate
num_epoch = args.num_epoch
batch_size = args.batch_size

if args.training_command is None:
    print(f"Select a training procedure: python cli.py {{{', '.join(get_args(TrainingType))}}}")
    sys.exit(0)


tokenizer_params = {
    k.removeprefix("tokenizer_"): v for k, v in args.__dict__.items() if k.startswith("tokenizer_") and k != "tokenizer_name"
}
elimination_params = {
    k.removeprefix("elimination_"): v for k, v in args.__dict__.items() if k.startswith("elimination_")
}

optimizer = STR2OPTIM[args.optimizer]
optimizer_params = {
    k.removeprefix(f"optimizer_{args.optimizer}_"): v
    for k, v in args.__dict__.items() if k.startswith(f"optimizer_{args.optimizer}_")
}
optimizer_params["lr"] = lr

lr_scheduler_params = {
    k.removeprefix("lr_scheduler_"): v for k, v in args.__dict__.items() if k.startswith("lr_scheduler_")
}

if args.use_wandb:
    import wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

loop = Loop(
    model_name=args.model_name,
    model_type=args.model_type,
    tokenizer_name=args.tokenizer_name,
    tokenizer_params=tokenizer_params,
    dataset_name=args.dataset_name,
    subset_name=args.subset_name,
    training_method=args.training_command,
    elimination=args.elimination,
    elimination_params=elimination_params,
    pre_evaluation=args.pre_evaluation,
    optimizer=optimizer,
    optimizer_params=optimizer_params
)

loop.train(
    lr_scheduler=args.lr_scheduler,
    lr_scheduler_params=lr_scheduler_params,
    batch_size=batch_size,
    num_epoch=num_epoch,
    logging_interval=args.logging_interval,
    use_wandb=args.use_wandb
)

loop.eval(use_wandb=args.use_wandb)
