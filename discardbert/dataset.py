from datasets import DatasetDict, load_dataset


def return_splits(dataset: str, subset: str) -> DatasetDict:
    num_labels = 2
    if dataset == "glue":
        ds = load_dataset(dataset, subset)
        val = ds['validation'].train_test_split(train_size=0.5, seed=0)
        ds['validation'] = val['train']
        ds['test'] = val['test']
        num_labels = len(self.dataset['train'].features['label'].names)
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
        num_labels = 3
    elif dataset == "conll":
        ds = load_dataset("conll2003")
        num_labels = ds['train'].features[f'{subset}_tags'].feature.names.__len__()
    else:
        ds = load_dataset(dataset, subset)
        try:
            num_labels = len(self.dataset['train'].features['label'].names)
        except:
            pass
    return ds, num_labels
