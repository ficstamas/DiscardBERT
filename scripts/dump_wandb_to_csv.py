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
        for art in artifacts:
            table = art
            if "progress_table" in table.qualified_name:
                break

        if table is None:
            continue

        table_dir = table.download()
        table_name = "progress_table"
        table_path = f"{table_dir}/{table_name}.table.json"
        with open(table_path) as file:
            json_dict = json.load(file)

        df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
        if "range" in df.columns:
            df[['range_from', 'range_to']] = pd.DataFrame(df.range.tolist(), index=df.index)
            df = df.drop("range", axis="columns")
        elif "pruned" in df.columns:
            pruned = df.pruned
            indices = [x for x in range(df.pruned.max()+1)]
            pruned_list = []
            for pr in pruned.tolist():
                pruned_list.append(indices[pr])
                for j in range(pr, len(indices)):
                    indices[j] -= 1
            df[['range_from', 'range_to']] = pd.DataFrame(
                {"range_from": pruned_list, "range_to": [x+1 for x in pruned_list]},
                index=df.index
            )
            df = df.drop("pruned", axis="columns")
        for i in range(len(df)):
            add_or_append("step", i)
            add_or_append("entity", run.entity)
            add_or_append("name", run.name)
            add_or_append("id", run.id)
            add_or_append("model", config["model_name"])
            add_or_append("dataset", config["dataset_name"])
            add_or_append("subset", config["subset_name"])
            add_or_append("initial_model", config["initial_model"] if "greedy" not in project else "pre-trained")
            add_or_append("project", config["wandb_project"] if "greedy" not in project else project)
            add_or_append("exit_condition", config["exit_condition"] if "greedy" not in project else "pre-trained")
            add_or_append("metric", config["target_metrics"] if "greedy" not in project else m2s[config["dataset_name"]][config["subset_name"]])
            add_or_append("seed", config["seed"])
        meta_df = pd.DataFrame(data=meta_data, index=df.index)
        output.append(pd.concat([meta_df, df], axis="columns"))

    output: pd.DataFrame = pd.concat(output, axis="rows", ignore_index=True)
    output.to_csv(f"recursive_{project}.csv")


extract_wandb("recursive")
extract_wandb("greedy-layer-pruning")
