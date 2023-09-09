import datetime
import os

from .elimination_strategy import EliminationType
from .training import STR2TRAINING, TrainingType
from .tokenizer import STR2TOKENIZER, DatasetType, SubsetType
from typing import Type, Dict
from transformers import get_scheduler
from torch.optim import Optimizer
from .dataset import return_splits
from .model.task_type import STR2MODEL_TYPE, ModelType
from .training.padding import STR2PADDING
from .metrics import STR2METRICS
import random
import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType


PERF2MODEL_TYPE = {
    "sequence": TaskType.SEQ_CLS,
    "token": TaskType.TOKEN_CLS
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Loop:
    def __init__(self, model_name: str, model_type: ModelType, tokenizer_name: str, tokenizer_params: dict,
                 dataset_name: DatasetType, subset_name: SubsetType,
                 training_method: TrainingType, trainer_params: Dict, elimination: EliminationType, elimination_params: dict,
                 pre_evaluation: bool, optimizer: Type[Optimizer] = None, optimizer_params: dict = None,
                 peft_params: dict = None,  seed: int = 42):
        set_seed(seed)

        if tokenizer_params is None:
            tokenizer_params = {}
        if optimizer_params is None:
            optimizer_params = {}
        if peft_params is None:
            perf_params = {}
        peft_params["task_type"] = PERF2MODEL_TYPE[model_type]
        peft_params["inference_mode"] = False

        self.model_type = model_type
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.training_method = training_method
        use_peft = peft_params.pop("peft")

        self.dataset = return_splits(dataset_name, subset_name)
        try:
            num_labels = len(self.dataset['train'].features['label'].names)
        except:
            num_labels = 2
        self.model = STR2MODEL_TYPE[model_type].from_pretrained(model_name, num_labels=num_labels)
        if use_peft:
            peft_config = LoraConfig(**peft_params)
            self.model = get_peft_model(self.model, peft_config)
        self.padding_fn = STR2PADDING[model_type]
        self.pre_evaluation = pre_evaluation
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.tokenizer = STR2TOKENIZER[dataset_name][subset_name](tokenizer_name, **tokenizer_params)
        self.tokenized_dataset = self.dataset.map(self.tokenizer.tokenize, batched=True)
        self.metrics = STR2METRICS[dataset_name][subset_name](
            dataset=dataset_name, subset=subset_name, id2label=self.model.config.id2label
        )

        trainer_params["compute_metrics"] = self.metrics.compute_metrics
        self.training = STR2TRAINING[training_method](
            self.model, self.tokenizer.tokenizer, elimination, elimination_params, **trainer_params
        )

    def train(self, lr_scheduler: str, lr_scheduler_params: dict, batch_size: int, num_epoch: int,
              logging_interval: int, use_wandb: bool, **kwargs):
        lr_scheduler_params["optimizer"] = self.optimizer

        if lr_scheduler_params["num_training_steps"] is None:
            lr_scheduler_params["num_training_steps"] = len(self.dataset["train"]) // batch_size * num_epoch

        scheduler = get_scheduler(lr_scheduler, **lr_scheduler_params)

        if self.pre_evaluation:
            self.eval(prefix="pre-eval", use_wandb=use_wandb)

        self.training.train(self.optimizer, scheduler, self.tokenized_dataset, self.padding_fn,
                            batch_size, num_epoch, logging_interval, use_wandb, **kwargs)
        path = f"experiments/{str(int(datetime.datetime.now().timestamp()))}/" \
               f"{self.model.config.name_or_path.replace('/', '_')}/{self.model_type}/" \
               f"{self.dataset_name}/{self.subset_name}/{self.training_method}/{self.training.path_information()}"
        os.makedirs(path, exist_ok=True)
        self.training.save(path, use_wandb)

    def eval(self, prefix="eval", use_wandb=False):
        self.training.eval(self.tokenized_dataset, self.metrics.compute_metrics, prefix=prefix, use_wandb=use_wandb)
