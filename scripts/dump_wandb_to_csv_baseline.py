import os
import wandb
import pandas as pd
import json
import tqdm


api = wandb.Api()
# entity, project = "szegedai-semantics", "greedy-layer-pruning"  # "recursive"  #

m2s = {
    "glue": {
        "cola": "matthews_correlation",
        "rte": "accuracy",
        "mrpc": "f1"
    },
    "conll2003": {
        "ner": "f1",
        "pos": "f1"
    }
}


def extract_wandb(project, entity: str = "szegedai-semantics"):
    runs = api.runs(entity + "/" + project)
    meta_data = {}

    def add_or_append(key, value):
        if key not in meta_data:
            meta_data[key] = [value]
        else:
            meta_data[key].append(value)

    output = []

    for run in tqdm.tqdm(runs):
        # if the key contains `_` then skip it (wandb internal variable)
        # if the type is `list` or `dict` then skip it (if it is important then have to flatten it manually)
        config = {
            k: v for k, v in run.config.items() if not k.startswith('_') and type(v) is not dict and type(v) is not list
        }
        meta_data = {}
        artifacts = run.logged_artifacts()
        table = None
        table_name = "metrics"
        for art in artifacts:
            table = art
            if table_name in table.qualified_name:
                break

        if table is None:
            continue

        table_dir = table.download()
        table_path = f"{table_dir}/{table_name}.table.json"
        with open(table_path) as file:
            json_dict = json.load(file)

        df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
        try:
            score = df[df['split'] == "test"].iloc[0].score
        except KeyError:
            continue
        add_or_append("entity", run.entity)
        add_or_append("name", run.name)
        add_or_append("id", run.id)
        add_or_append("model", config["model_name"])
        add_or_append("dataset", config["dataset_name"])
        add_or_append("subset", config["subset_name"])
        add_or_append("initial_model", config["initial_model"] if "greedy" not in project else "pre-trained")
        add_or_append("metric", config["target_metrics"] if "greedy" not in project else m2s[config["dataset_name"]][config["subset_name"]])
        add_or_append("seed", config["seed"])
        add_or_append("score", score)
        meta_df = pd.DataFrame(data=meta_data, index=[0])
        output.append(meta_df)

    output: pd.DataFrame = pd.concat(output, axis="rows", ignore_index=True)
    output.to_csv(f"baseline.csv")


extract_wandb("recursive")

