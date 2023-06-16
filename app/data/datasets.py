import os
import ast
import json
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader


def load_json_data(data_dir: str) -> pd.DataFrame:
    """
    Load all jsonl files in the data directory and returns a dataframe
    Returns: pd.DataFrame containing the data
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('jsonl')]
    rows = []
    for f in files:
        with open(f'{data_dir}/{f}', 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            label = result["label"]
            text = result["text"]
            new_spans = []
            if result["spans"] and len(result["spans"]) != 0:
                try:
                    new_spans = [s["text"] for s in result["spans"]]
                except:
                    x = ast.literal_eval(result["spans"])
                    new_spans = [s["text"] for s in x]
            new_row = {"text": text, "label": label, "spans": new_spans}
            rows.append(new_row)
        print(f"Loaded: {f}")
    df = pd.DataFrame(rows)
    return df


class TAndIDataSet(Dataset):
    """
    Custom PyTorch Dataset for training hugging face transformer models
    Args:
    data: the data to encode
    tokenizer: the tokenizer to vectorize the text data
    label_encoder: the label encoder for a numeric representation of the labels
    """
    encodings: torch.Tensor
    encoded_labels: torch.Tensor

    def __init__(self, data: pd.DataFrame, tokenizer, label_encoder):
        self.data = data
        self.label_encoder = label_encoder
        self._encode(tokenizer, label_encoder)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.encoded_labels[idx])
        return item

    def __len__(self):
        return len(self.data)

    def _encode(self, tokenizer, label_encoder):
        self.encodings = tokenizer(self.data.text.tolist(), truncation=True, padding=True, max_length=512)
        self.encoded_labels = label_encoder.fit_transform(self.data.label.tolist())


def get_data_loaders(data_path:str, model, batch_sizes:dict):
    """
    given a data-path, a model, and the batch-sizes, return a ready to use dictionary of data loaders for training
    hugging face transformer models
    Args:
        data_path: the path to the data
        model: the transformer used for classification
        batch_sizes: the batch sizes for the train, val, and test data loaders
    Returns: a dictionary of data loaders for train, val, and test
    """
    # load csv/json
    df = load_json_data(data_path)
    # df = df[:1000]
    # get encodings for labels
    le = preprocessing.LabelEncoder()
    le.fit(df.label)
    # split into train, test, val
    # TODO: make split sizes configurable
    train_df, val_df, test_df = np.split(df.sample(frac=1), [int(.7 * len(df)), int(.9 * len(df))])
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    datasets = {
        "train": DataLoader(TAndIDataSet(train_df, model.tokenizer, le), batch_size=batch_sizes["train"], shuffle=True),
        "val": DataLoader(TAndIDataSet(val_df, model.tokenizer, le), batch_size=batch_sizes["val"], shuffle=True),
        "test": DataLoader(TAndIDataSet(test_df, model.tokenizer, le), batch_size=batch_sizes["test"], shuffle=True)}
    return datasets
