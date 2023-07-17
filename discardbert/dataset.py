from datasets import DatasetDict, load_dataset


def return_splits(dataset: str, subset: str) -> DatasetDict:
    if dataset == "glue":
        ds = load_dataset(dataset, subset)
        val = ds['validation'].train_test_split(train_size=0.7, seed=0)
        ds['validation'] = val['train']
        ds['test'] = val['test']
    else:
        ds = load_dataset(dataset, subset)
    return ds
