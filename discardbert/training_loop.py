from .elimination_strategy import STR2ELIMINATION, EliminationType
from .training import STR2TRAINING, TrainingType
from .tokenizer import STR2TOKENIZER, DatasetType, SubsetType
from typing import Type
from transformers import get_scheduler
from torch.optim import Optimizer
from datasets import load_dataset
from .model.task_type import STR2MODEL_TYPE, ModelType
from .training.padding import STR2PADDING
from .metrics import STR2METRICS


class Loop:
    def __init__(self, model_name: str, model_type: ModelType, tokenizer_name: str, tokenizer_params: dict,
                 dataset_name: DatasetType, subset_name: SubsetType,
                 training_method: TrainingType, elimination: EliminationType, elimination_params: dict,
                 pre_evaluation: bool, optimizer: Type[Optimizer] = None, optimizer_params: dict = None,):
        if tokenizer_params is None:
            tokenizer_params = {}
        if optimizer_params is None:
            optimizer_params = {}

        self.model = STR2MODEL_TYPE[model_type].from_pretrained(model_name)
        self.padding_fn = STR2PADDING[model_type]
        self.pre_evaluation = pre_evaluation
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.tokenizer = STR2TOKENIZER[dataset_name][subset_name](tokenizer_name, **tokenizer_params)
        self.dataset = load_dataset(dataset_name, subset_name)
        self.tokenized_dataset = self.dataset.map(self.tokenizer.tokenize, batched=True)
        self.metrics = STR2METRICS[dataset_name][subset_name](dataset=dataset_name, subset=subset_name)
        self.training = STR2TRAINING[training_method](
            self.model, self.tokenizer.tokenizer, elimination, **elimination_params
        )

    def train(self, lr_scheduler: str, lr_scheduler_params: dict, batch_size: int, num_epoch: int,
              logging_interval: int, use_wandb: bool, **kwargs):
        lr_scheduler_params["optimizer"] = self.optimizer
        scheduler = get_scheduler(lr_scheduler, **lr_scheduler_params)

        if self.pre_evaluation:
            self.eval(prefix="pre-eval")

        self.training.train(self.optimizer, scheduler, self.tokenized_dataset["train"], self.padding_fn,
                            batch_size, num_epoch, logging_interval, use_wandb, **kwargs)

    def eval(self, prefix="eval"):
        self.training.eval(self.tokenized_dataset, self.metrics.compute_metrics, prefix=prefix)
