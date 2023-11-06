import os
import wandb
import pandas as pd
import json


api = wandb.Api()
entity, project = "szegedai-semantics", "recursive"
runs = api.runs(entity + "/" + project)

meta_data = {}


def add_or_append(key, value):
    if key not in meta_data:
        meta_data[key] = [value]
    else:
        meta_data[key].append(value)


output = []

for run in runs:
    # if the key contains `_` then skip it (wandb internal variable)
    # if the type is `list` or `dict` then skip it (if it is important then have to flatten it manually)
    config = {
        k: v for k, v in run.config.items() if not k.startswith('_') and type(v) is not dict and type(v) is not list
    }
    meta_data = {}
    artifacts = run.logged_artifacts()
    for art in artifacts:
        table = art
        if "progress_table" in table.qualified_name:
            break

    if table is None:
        print("Fuck!")
        exit(2)

    table_dir = table.download()
    table_name = "progress_table"
    table_path = f"{table_dir}/{table_name}.table.json"
    with open(table_path) as file:
        json_dict = json.load(file)

    df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
    df[['range_from', 'range_to']] = pd.DataFrame(df.range.tolist(), index=df.index)
    df = df.drop("range", axis="columns")

    for i in range(len(df)):
        add_or_append("step", i)
        add_or_append("entity", run.entity)
        add_or_append("name", run.name)
        add_or_append("id", run.id)
        add_or_append("model", config["model_name"])
        add_or_append("dataset", config["dataset_name"])
        add_or_append("subset", config["subset_name"])
        add_or_append("initial_model", config["initial_model"])
        add_or_append("project", config["wandb_project"])
        add_or_append("exit_condition", config["exit_condition"])
        add_or_append("metric", config["target_metrics"])
        add_or_append("seed", config["seed"])
    meta_df = pd.DataFrame(data=meta_data, index=df.index)
    output.append(pd.concat([meta_df, df], axis="columns"))

output: pd.DataFrame = pd.concat(output, axis="rows", ignore_index=True)
output.to_csv("recursive.csv")
