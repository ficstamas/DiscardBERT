import copy
import json
import os.path
from typing import Optional, Callable

import torch
from datasets import DatasetDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .base import PreTrainedModel, PreTrainedTokenizer, EliminationType, Dict
from .simple import Simple
import numpy as np
import pandas as pd
from .generators.steps import STEP_GENERATOR


class Recursive(Simple):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 elimination_type: EliminationType, elimination_params: Dict, exit_params: Dict,
                 compute_metrics=None, target_metrics="val", **kwargs):
        super().__init__(model, tokenizer, "range", elimination_params)
        self.exit_params = exit_params
        self.dilation_step = kwargs.get("dilation_step", 1)
        self.exploration_function = STEP_GENERATOR[
            kwargs.get("recursive_steps", "full_triangle")
        ]
        num_layers = model.config.num_hidden_layers
        max_depth = self.exit_params["max_depth"] \
            if self.exit_params["max_depth"] != -1 \
            else num_layers // self.dilation_step + 1
        self.metrics = {
            "train": np.ones((max_depth, num_layers, num_layers + 1)) * -np.inf,
            "dev":   np.ones((max_depth, num_layers, num_layers + 1)) * -np.inf,
            "test":  np.ones((max_depth, num_layers, num_layers + 1)) * -np.inf
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
        # run pdf or pfdf
        initial_model = kwargs.get("initial_model", "pdf")

        if initial_model == "pfdf":
            trainer = Simple(self.model, self.tokenizer, "range", {"range": (0, 0)})
            trainer.train(optimizer.__class__(self.model.parameters(), **optimizer.defaults),
                          copy.deepcopy(lr_scheduler), dataset, padding_fn,
                          batch_size, num_epoch, logging_interval, use_wandb, elimination_applied=True, **kwargs)
            self.model = trainer.model
            del trainer

        # get baseline score
        prefix = "pre-eval"
        baseline = self.eval(dataset=dataset, compute_metrics=self.compute_metrics, prefix=prefix)
        for key in baseline:
            self.metrics[key][0, 0, 0] = baseline[key][f"{prefix}_{key}_{self.target_metrics}"]
            if use_wandb:
                import wandb
                wandb.log({f"progress_{key}": self.metrics[key][0, 0, 0].item()})

        model = copy.deepcopy(self.model)
        depth = 1
        if use_wandb:
            import wandb
            wandb.log({
                "num_layers": model.config.num_hidden_layers
            })
        # iterate depth
        while True:
            # iterate possible values
            num_layers = model.config.num_hidden_layers
            for i, j in self.exploration_function(0, num_layers, self.dilation_step):
                # training a sub model
                trainer = Simple(copy.deepcopy(model), self.tokenizer, "range", {"range": (i, j)})
                trainer.apply_elimination()
                trainer.train(optimizer.__class__(trainer.model.parameters(), **optimizer.defaults),
                              copy.deepcopy(lr_scheduler), dataset, padding_fn,
                              batch_size, num_epoch, logging_interval, use_wandb, elimination_applied=True, **kwargs)
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
                metric_coordinates = (int(metric_coordinates[0][0]), int(metric_coordinates[1][0]))
            else:
                raise NotImplementedError()

            print(f"Coordinates for selection:", metric_coordinates)
            # logging
            for key in self.metrics:
                self.history[key]["score"].append(
                    self.metrics[key][depth, metric_coordinates[0], metric_coordinates[1]].item()
                )
                self.history[key]["range"].append(metric_coordinates)
                if use_wandb:
                    import wandb
                    wandb.log({
                        f"progress_{key}": self.metrics[key][depth, metric_coordinates[0], metric_coordinates[1]].item(),
                        "num_layers": num_layers - (metric_coordinates[1] - metric_coordinates[0])
                    })

            print(f"Num layers before: ", model.config.num_hidden_layers)
            # create new base model
            sub = Simple(copy.deepcopy(model), self.tokenizer, "range", {"range": metric_coordinates})
            sub.apply_elimination()
            sub.train(optimizer.__class__(sub.model.parameters(), **optimizer.defaults), copy.deepcopy(lr_scheduler),
                      dataset, padding_fn, batch_size, num_epoch, logging_interval, use_wandb, elimination_applied=True,
                      **kwargs)
            model = sub.model
            del sub
            print(f"Num layers after: ", model.config.num_hidden_layers)
            if use_wandb:
                import wandb
                wandb.log({
                    "num_layers": model.config.num_hidden_layers
                })

            depth += 1

            # stopping criteria
            # if we have no layers
            if model.config.num_hidden_layers == 0:
                break
            # if depth reached
            if depth == self.exit_params["max_depth"]:
                break
        self.model = model

    def save(self, path: str, use_wandb: bool = False):
        for key, val in self.metrics.items():
            np.save(os.path.join(path, f"metric_{key}.npy"), val)

        with open(os.path.join(path, "history.json"), mode="w") as f:
            json.dump(self.history, f)

        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

        if use_wandb:
            data = {
                k: v["score"] for k, v in self.history.items()
            }
            data["range"] = self.history["train"]["range"]
            df = pd.DataFrame(data=data)

            import wandb
            wandb.log({"progress_table": wandb.Table(dataframe=df)})

            data = {"split": [], "depth": [], "from": [], "to": [], "score": []}
            for split in self.metrics:
                arr = self.metrics[split]
                for depth in range(arr.shape[0]):
                    for from_ in range(arr.shape[1]):
                        for to_ in range(arr.shape[2]):
                            if not np.isinf(arr[depth, from_, to_]):
                                data["split"].append(split)
                                data["depth"].append(depth)
                                data["from"].append(from_)
                                data["to"].append(to_)
                                data["score"].append(arr[depth, from_, to_])
            df = pd.DataFrame(data=data)
            wandb.log({"metrics": wandb.Table(dataframe=df)})

    def path_information(self) -> str:
        repr_ = []
        for key, value in self.exit_params.items():
            repr_.append(f"{key}-{value}")
        repr_.append(f"dilation_step-{self.dilation_step}")
        return f"{self.elimination.path_information(**self.elimination_params)}/{'_'.join(repr_)}"
