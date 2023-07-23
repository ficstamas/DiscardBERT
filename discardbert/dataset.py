from datasets import DatasetDict, load_dataset


def return_splits(dataset: str, subset: str) -> DatasetDict:
    if dataset == "glue":
        ds = load_dataset(dataset, subset)
        val = ds['validation'].train_test_split(train_size=0.5, seed=0)
        ds['validation'] = val['train']
        ds['test'] = val['test']
    elif dataset == "wanli":
        ds = load_dataset("alisawuffles/WANLI")
        # rename 'gold' to 'label'
        train = ds['train'].rename_columns({"gold": "label"}).class_encode_column("label")
        test = ds['test'].rename_columns({"gold": "label"}).class_encode_column("label")
        ds = DatasetDict({
            "train": train,
            "test": test
        })

        val = ds['test'].train_test_split(train_size=0.5, seed=0)
        ds['validation'] = val['train']
        ds['test'] = val['test']
    else:
        ds = load_dataset(dataset, subset)
    return ds
