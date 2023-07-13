import copy
import json
import os.path
from typing import Optional, Callable

from datasets import Dataset, DatasetDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .base import PreTrainedModel, PreTrainedTokenizer, EliminationType, Dict
from .simple import Simple
import numpy as np


class Recursive(Simple):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 elimination_type: EliminationType, elimination_params: Dict, exit_params: Dict,
                 compute_metrics=None, target_metrics="val", **kwargs):
        super().__init__(model, tokenizer, "range", elimination_params)
        self.exit_params = exit_params
        self.dilation_step = kwargs.get("dilation_step", 1)
        num_layers = model.config.num_hidden_layers
        max_depth = self.exit_params["max_depth"] \
            if self.exit_params["max_depth"] != -1 \
            else num_layers // self.dilation_step + 1
        self.metrics = {
            "train": np.zeros((max_depth, num_layers, num_layers)) * np.inf,
            "dev":   np.zeros((max_depth, num_layers, num_layers)) * np.inf,
            "test":  np.zeros((max_depth, num_layers, num_layers)) * np.inf
        }
        self.compute_metrics = compute_metrics
        self.target_metrics = target_metrics
        self.history = {
            "train": {"score": [], "range": []},
            "dev": {"score": [], "range": []},
            "test": {"score": [], "range": []},
        }

    def train(self, optimizer: Optimizer, lr_scheduler: LRScheduler, dataset: DatasetDict, padding_fn: Optional[Callable],
              batch_size: int, num_epoch: int, logging_interval: int, use_wandb: bool, **kwargs):
        # get baseline score
        prefix = "pre-eval"
        baseline = self.eval(dataset=dataset, compute_metrics=self.compute_metrics, prefix=prefix)
        for key in baseline:
            self.metrics[key][0, 0, 0] = baseline[key][f"{prefix}_{key}_{self.target_metrics}"]

        model = copy.deepcopy(self.model)
        depth = 1
        # iterate depth
        while True:
            # iterate possible values
            num_layers = model.config.num_hidden_layers
            for i in range(0, num_layers, self.dilation_step):
                for j in range(i+self.dilation_step, num_layers, self.dilation_step):
                    # training a sub model
                    trainer = Simple(copy.deepcopy(model), self.tokenizer, "range", {"range": (i, j)})
                    trainer.train(optimizer, lr_scheduler, dataset, padding_fn, batch_size, num_epoch, logging_interval,
                                  use_wandb, **kwargs)
                    prefix = f"eval_{depth}_{i}_{j}"
                    metrics = trainer.eval(dataset, self.compute_metrics, prefix)
                    for key in baseline:
                        self.metrics[key][depth, i, j] = metrics[key][f"{prefix}_{key}_{self.target_metrics}"]
                    del trainer

            # selecting best elimination
            depth_metric = self.metrics["dev"][depth, :num_layers, :num_layers]
            if self.exit_params["selection_criteria"] == "best":
                metric = np.max(depth_metric)
                if np.isinf(metric):
                    break

                metric_coordinates = np.where(depth_metric == metric)
                metric_coordinates = (metric_coordinates[0].item(), metric_coordinates[1].item())
            else:
                raise NotImplementedError()

            # logging
            for key in self.metrics:
                self.history[key]["score"].append(
                    self.metrics[key][depth, metric_coordinates[0], metric_coordinates[1]].item()
                )
                self.history[key]["range"].append(metric_coordinates)
                if use_wandb:
                    import wandb
                    wandb.log({
                        f"progress_{key}": self.metrics[key][depth, metric_coordinates[0], metric_coordinates[1]].item()
                        for key in self.metrics
                    })

            # create new base model
            sub = Simple(copy.deepcopy(model), self.tokenizer, "range", {"range": metric_coordinates})
            sub.apply_elimination()
            model = sub.model
            del sub

            depth += 1

            # stopping criteria
            # if we have no layers
            if model.config.num_hidden_layers == 0:
                break
            # if depth reached
            if depth == self.exit_params["max_depth"]:
                break

    def save(self, path: str):
        for key, val in self.metrics.items():
            np.save(os.path.join(path, f"metric_{key}.npy"), val)

        with open(os.path.join(path, "history.json"), mode="w") as f:
            json.dump(self.history, f)
