import sys
from argparse import ArgumentParser
from discardbert.training_loop import Loop, ModelType, DatasetType, SubsetType, TrainingType, EliminationType
from typing import get_args, get_origin, Union, Literal, Iterable
from discardbert.optimizer import STR2OPTIM, OptimType


def flatten_type(type_args) -> Iterable[str]:
    """
    Get defined arguments in Union of Literals or in Literals
    :param type_args:
    :return:
    """
    flattened_type_args = ()
    if get_origin(type_args) is Union:
        for type_ in get_args(type_args):
            flattened_type_args += get_args(type_)
    else:
        flattened_type_args += get_args(type_args)
    return flattened_type_args


args_ = ArgumentParser("DiscardBERT")

# general parameters
args_.add_argument("--model_name", type=str, default="prajjwal1/bert-medium")
args_.add_argument("--model_type", type=str, choices=flatten_type(ModelType), default="sequence")
args_.add_argument("--dataset_name", type=str, choices=flatten_type(DatasetType), default="glue")
args_.add_argument("--subset_name", type=str, choices=flatten_type(SubsetType), default="mrpc")
args_.add_argument("--pre_evaluation", action="store_true")
args_.add_argument("--initial_model", type=str, choices=["pdf", "pfdf"], default="pdf")

args_.add_argument("--learning_rate", type=float, default=2e-5)
args_.add_argument("--num_epoch", type=int, default=3)
args_.add_argument("--batch_size", type=int, default=16)
args_.add_argument("--seed", type=int, default=42)
args_.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cpu")

# optimizer parameters
args_.add_argument("--optimizer", type=str, choices=flatten_type(OptimType), default="adamw")
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
args_.add_argument("--elimination", type=str, choices=flatten_type(EliminationType))
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
recursive.add_argument("--exit_condition", default="max_depth", choices=["depth", "tolerance"], help="Condition of stopping")
recursive.add_argument("--selection_criteria", default="best", choices=["best"], help="How to select the next sub-model")
recursive.add_argument("--max_depth", default=-1, type=int, help="Exit after reaching maximum depth")
recursive.add_argument("--max_tolerance", default=0.95, type=float, help="Exit after tolerance reached")
recursive.add_argument("--dilation_step", default=1, type=int, help="How many blocks of layers to jump")
recursive.add_argument("--target_metrics", default="f1", help="Metric to use for validation")

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
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

trainer_params = {
    "exit_params": {
        "exit_condition": args.exit_condition,
        "max_depth": args.max_depth,
        "max_tolerance": args.max_tolerance,
        "selection_criteria": args.selection_criteria
    },
    "target_metrics": args.target_metrics,
    "dilation_step": args.dilation_step
}

loop = Loop(
    model_name=args.model_name,
    model_type=args.model_type,
    tokenizer_name=args.tokenizer_name,
    tokenizer_params=tokenizer_params,
    dataset_name=args.dataset_name,
    subset_name=args.subset_name,
    training_method=args.training_command,
    trainer_params=trainer_params,
    elimination=args.elimination,
    elimination_params=elimination_params,
    pre_evaluation=args.pre_evaluation,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    seed=args.seed
)

loop.train(
    lr_scheduler=args.lr_scheduler,
    lr_scheduler_params=lr_scheduler_params,
    batch_size=batch_size,
    num_epoch=num_epoch,
    logging_interval=args.logging_interval,
    use_wandb=args.use_wandb,
    initial_model=args.initial_model,
    device=args.device
)

loop.eval(use_wandb=args.use_wandb)
