import os
import ast
import json
import torch
import itertools
import pandas as pd
from pickle import dump
from sklearn import preprocessing
from torch import Generator
from transformers import AutoTokenizer, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from training.utils.utils import logger

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


def load_json_data(data_dir: str) -> pd.DataFrame:
    """
    Load all jsonl files in the data directory and returns a dataframe
    :param data_dir: the path to the directory containing the json files (one per clas)
    :return: the data as a pd.Dataframe
    """
    data_dir = os.path.join(data_dir, "annotated_data")
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
            new_row = {
                "text": text,
                "label": label,
                "spans": new_spans,
                "article_id": result["meta"]["doc_id"],
                "sentence_id": result["meta"]["sent_id"]
            }
            rows.append(new_row)
        logger.info(f"Loaded: {f}")
    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["article_id", "sentence_id"], inplace=True)
    return df


def encode_labels(df, config):
    le = preprocessing.LabelEncoder()
    le.fit(df.label)
    if not os.path.exists(config["checkpoints_dir"]):
        os.makedirs(config["checkpoints_dir"])
    # save label encoder
    dump(le, open(os.path.join(config["checkpoints_dir"], "label_encoder.pkl"), 'wb'))
    return df, le


def split_data_into_train_val_test(df, config):
    label_counts = df.label.value_counts()
    logger.info(f"Number of classes: {len(label_counts)}")
    labels_with_less_than_ten_samples = label_counts[label_counts <= 10]
    logger.info(f"Labels with less than 10 samples: {labels_with_less_than_ten_samples}")
    df = df[~df["label"].isin(labels_with_less_than_ten_samples.index)]
    config["num_labels"] = len(df.label.unique())
    train_size = int(config["dataset_splits"][0] * len(df))
    val_size = int((config["dataset_splits"][1] - config["dataset_splits"][0]) * len(df))
    test_size = len(df) - train_size - val_size
    train_df, temp_df = train_test_split(df, test_size=val_size + test_size, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size, stratify=temp_df['label'], random_state=42)
    return train_df, val_df, test_df


def get_data_loaders_with_chatgpt_annotated_data(config, debug=False, inference=False):
    df = pd.read_parquet(os.path.join(config["data_dir"], "openai_annotated_data.parquet"))
    # TODO: this should not be here
    df.loc[df["label"] == "3d_printed_apparel", "label"] = "3d_printed_clothes"
    if debug:
        # sample from every label to get a small dataset
        dfs = []
        for label in df.label.unique():
            num_samples_for_label = len(df[df["label"] == label])
            dfs.append(df[df["label"] == label].sample(min(10, num_samples_for_label)))
        df = pd.concat(dfs)
    # get encodings for labels
    df, le = encode_labels(df, config)
    # split into train, test, val
    train_df, val_df, test_df = split_data_into_train_val_test(df, config)
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    tokenizer = init_tokenizer(train_df, config)
    logger.info(f"Size of train_df: {len(train_df)}")
    logger.info(f"Size of val_df: {len(val_df)}")
    logger.info(f"Size of test_df: {len(test_df)}")
    datasets = {"train": DataLoader(TAndIDataSet(train_df, tokenizer, le), batch_size=config["train_batch_size"],
                                    shuffle=True, generator=Generator().manual_seed(2147483647)),
        "val": DataLoader(TAndIDataSet(val_df, tokenizer, le), batch_size=config["val_batch_size"], shuffle=True,
                          generator=Generator().manual_seed(2147483647)),
        "test": DataLoader(TAndIDataSet(test_df, tokenizer, le), batch_size=config["test_batch_size"], shuffle=True,
                           generator=Generator().manual_seed(2147483647))}
    if not inference:
        tokenizer.save_pretrained(config['checkpoints_dir'])
    return datasets, le, tokenizer


def get_data_loaders(config, debug=False):
    """
    Given a data-path, a model, and the batch-sizes, return a ready to use dictionary of data loaders for training
    hugging face transformer models
    :param config: the training config
    :return: a dictionary of data loaders for train, val, and test
    """
    # load csv/json
    df = load_json_data(config["data_dir"])
    # debug mode
    if debug:
        # sample from every label to get a small dataset
        dfs = []
        for label in df.label.unique():
            dfs.append(df[df["label"] == label].sample(10))
        df = pd.concat(dfs)
    # get encodings for labels
    le = preprocessing.LabelEncoder()
    le.fit(df.label)
    # save label encoder
    # create directory if it does not exist
    if not os.path.exists(config["checkpoints_dir"]):
        os.makedirs(config["checkpoints_dir"])
    dump(le, open(os.path.join(config["checkpoints_dir"], "label_encoder.pkl"), 'wb'))
    # split into train, test, val
    train_df, val_df, test_df = split_data_into_train_val_test(df, config)
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    tokenizer = init_tokenizer(train_df, config)
    logger.info(f"Size of train_df: {len(train_df)}")
    logger.info(f"Size of val_df: {len(val_df)}")
    logger.info(f"Size of test_df: {len(test_df)}")
    datasets = {
        "train": DataLoader(TAndIDataSet(train_df, tokenizer, le), batch_size=config["train_batch_size"], shuffle=True,
                            generator=Generator().manual_seed(2147483647)),
        "val": DataLoader(TAndIDataSet(val_df, tokenizer, le), batch_size=config["val_batch_size"], shuffle=True,
                          generator=Generator().manual_seed(2147483647)),
        "test": DataLoader(TAndIDataSet(test_df, tokenizer, le), batch_size=config["test_batch_size"], shuffle=True,
                           generator=Generator().manual_seed(2147483647))}
    tokenizer.save_pretrained(config['checkpoints_dir'])
    return datasets, le, tokenizer


def init_tokenizer(train_df, config):
    if config["model_name"] == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", use_fast=False, max_len=512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False, max_len=512)
        new_spans = set(itertools.chain.from_iterable(train_df.spans.tolist()))
        new_tokens = list(new_spans - set(tokenizer.vocab.keys()))
        tokenizer.add_tokens(new_tokens)
    return tokenizer


def get_data_loader(dataset:str):
    if dataset == "old_data":
        return get_data_loaders
    elif dataset == "openai_annotated_data":
        return get_data_loaders_with_chatgpt_annotated_data
    else:
        raise ValueError(f"Unknown dataset: {dataset}")