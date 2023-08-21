import os
import ast
import json
import torch
import numpy as np
import pandas as pd
from pickle import dump
from sklearn import preprocessing
from torch import Generator
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


def load_json_data(data_dir: str) -> pd.DataFrame:
    """
    Load all jsonl files in the data directory and returns a dataframe
    :param data_dir: the path to the directory containing the json files (one per clas)
    :return: the data as a pd.Dataframe
    """
    rows = []
    for f in [f for f in os.listdir(data_dir) if f.endswith('jsonl')]:
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
    encodings: torch.Tensor
    encoded_labels: torch.Tensor

    def __init__(self, data: pd.DataFrame, tokenizer, label_encoder):
        """
        Custom PyTorch Dataset for training hugging face transformer models for the TandI usecase
        :param data: the data to encode
        :param tokenizer: the tokenizer to vectorize the text data
        :param label_encoder: the label encoder for a numeric representation of the labels
        """
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


def get_data_loaders(config):
    """
    Given a data-path, a model, and the batch-sizes, return a ready to use dictionary of data loaders for training
    hugging face transformer models
    :param config: the training config
    :return: a dictionary of data loaders for train, val, and test
    """
    # load csv/json
    df = load_json_data(config["data_dir"])
    df = df.sample(n=100, random_state=42)
    # get encodings for labels
    le = preprocessing.LabelEncoder()
    le.fit(df.label)
    # save label encoder
    # create directory if it does not exist
    if not os.path.exists(config["save_model_dir"]):
        os.makedirs(config["save_model_dir"])
    dump(le, open(os.path.join(config["save_model_dir"], "label_encoder.pkl"), 'wb'))
    # split into train, test, val
    train_df, val_df, test_df = np.split(
        df.sample(frac=1, random_state=42),
        [int(config["dataset_splits"][0] * len(df)), int(config["dataset_splits"][1] * len(df))]
    )
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    datasets = {
        "train": DataLoader(TAndIDataSet(train_df, tokenizer, le), batch_size=config["batch_sizes"]["train"], shuffle=True,
                            generator=Generator().manual_seed(2147483647)),
        "val": DataLoader(TAndIDataSet(val_df, tokenizer, le), batch_size=config["batch_sizes"]["val"], shuffle=True,
                          generator=Generator().manual_seed(2147483647)),
        "test": DataLoader(TAndIDataSet(test_df, tokenizer, le), batch_size=config["batch_sizes"]["test"], shuffle=True,
                           generator=Generator().manual_seed(2147483647))}
    tokenizer.save_pretrained(config['save_model_dir'])
    return datasets
