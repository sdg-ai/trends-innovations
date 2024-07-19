import os
import ast
import json
import torch
import pandas as pd
from pickle import dump
from sklearn import preprocessing
from torch import Generator
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
import os
import pandas as pd
import itertools
from typing import Tuple
from sklearn import preprocessing
from pickle import dump
from transformers import AutoTokenizer, RobertaTokenizer
from utils.utils import logger as logger

import os
import pandas as pd
import itertools
from typing import Tuple
from sklearn import preprocessing
from pickle import dump
from transformers import AutoTokenizer, RobertaTokenizer

human_categories_to_chatgpt_categories = {
    "3d_printed_clothes": "3d_printed_apparel",
}

categories_to_combine = {
    "energy": ["solar_energy","solar_energy","solar_energy", "hydropower","energy_storage"],
    "sustainable_clothing": ["3d_printed_apparel", "capsule_wardrobe", "clothes_designed_for_a_circular_economy", "rent_apparel"],
    "sharing_economy": ["sharing_economy", "car_sharing"]
}


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


def encode_labels(df, checkpoints_dir:str) -> Tuple[pd.DataFrame, preprocessing.LabelEncoder]:
    le = preprocessing.LabelEncoder()
    le.fit(df.label)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    dump(
        le,
        open(os.path.join(checkpoints_dir, "label_encoder.pkl"),
             'wb'))
    return df, le


def init_tokenizer(train_df:pd.DataFrame, model_name:str) -> AutoTokenizer:
    if model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", use_fast=False, max_len=512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, max_len=512)
        new_spans = set(itertools.chain.from_iterable(train_df.spans.tolist()))
        new_tokens = list(new_spans - set(tokenizer.vocab.keys()))
        tokenizer.add_tokens(new_tokens)
    return tokenizer


def split_data_into_train_val_test(
        df:pd.DataFrame, 
        train_size: float,
        val_size: float,
        test_size: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_training_samples = int(train_size * len(df))
    num_val_samples = int(val_size * len(df))
    num_test_samples = len(df) - num_training_samples - num_val_samples
    train_df, temp_df = train_test_split(df, test_size=num_val_samples + num_test_samples, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=num_test_samples, stratify=temp_df['label'], random_state=42)
    return train_df, val_df, test_df


def load_human_annotated_data(
        keep_ignores: bool,
        keep_rejects: bool,
        label_reject_and_ignore_as_irrelevant: bool,
        drop_conflicting_answers: bool
) -> pd.DataFrame:
    logger.info("Loading human annotated data.")
    if label_reject_and_ignore_as_irrelevant and (keep_ignores or keep_rejects):
        raise ValueError(
            "Cannot label 'reject' and 'ignore' as 'irrelevant' while keeping 'reject' or 'ignore' labels."
        )
    data_dir = "datasets/annotated_data"
    rows = []
    for f in [f for f in os.listdir(data_dir) if f.endswith('jsonl')]:
        with open(f'{data_dir}/{f}', 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            raw_row = json.loads(json_str)
            new_spans = []
            if raw_row["spans"] and len(raw_row["spans"]) != 0:
                if isinstance(raw_row["spans"], str):
                    raw_row["spans"] = raw_row["spans"].replace("'", '"')
                    spans = json.loads(raw_row["spans"])
                else:
                    spans = raw_row["spans"]
                new_spans = [s["text"] for s in spans]
            new_row = {
                "text": raw_row["text"],
                "label": raw_row["label"],
                "spans": new_spans,
                "article_id": raw_row["meta"]["doc_id"],
                "section_id": raw_row["meta"]["sent_id"],
                "answer": raw_row["answer"]
            }
            rows.append(new_row)
    df = pd.DataFrame(rows)
    if drop_conflicting_answers:
        unique_duplicate_article_section_ids = df[df.duplicated(subset=["article_id", "section_id"])][["article_id", "section_id"]].drop_duplicates()
        num_dropped = 0
        for _, row in unique_duplicate_article_section_ids.iterrows():
            duplicate_rows = df[(df["article_id"] == row["article_id"]) & (df["section_id"] == row["section_id"])]
            if len(duplicate_rows["answer"].unique()) > 1:
                df.drop(duplicate_rows.index, inplace=True)
                num_dropped += len(duplicate_rows) 
        logger.info(f"Dropped {num_dropped} rows due to conflicting answers from human annotators.")
    len_before = len(df)
    df = df.drop_duplicates(subset=["article_id", "section_id"])
    logger.info(f"Dropped {len_before-len(df)} duplicates from agreeing human annotated data.")
    if label_reject_and_ignore_as_irrelevant:
        logger.info("Labeling answers 'reject' and 'ignore' as 'irrelevant' in human annotated data.")
        df.loc[df["answer"].isin(["reject", "ignore"]), "label"] = "irrelevant"
    if not keep_ignores and not label_reject_and_ignore_as_irrelevant:
        df = df.drop(df[df["answer"] == "ignore"].index)
    if not keep_rejects and not label_reject_and_ignore_as_irrelevant:
        df = df.drop(df[df["answer"] == "reject"].index)
    df.drop(columns=["answer"], inplace=True)
    logger.info(f"Loaded total of {len(df)} samples from human annotated data.")
    return df


def load_chatgpt_annotated_data() -> pd.DataFrame:
    logger.info("Loading ChatGPT annotated data.")
    df = pd.read_parquet(os.path.join("datasets", "openai_annotated_data.parquet"))
    logger.info(f"Loaded total of {len(df)} samples from ChatGPT annotated data.")
    return df


def merge_categories(df: pd.DataFrame) -> pd.DataFrame:
    for new_category, categories in categories_to_combine.items():
        for category in categories:
            df.loc[df["label"] == category, "label"] = new_category
    return df


def load_data(
        min_samples_per_label: int,
        use_human_annotated_data: bool,
        use_chatgpt_annotated_data: bool,
        undersample: bool,
        upsample: bool,
        combine_categories:bool,
        debug:bool,
        **kwargs
) ->  pd.DataFrame:
    if use_human_annotated_data:
        df_human = load_human_annotated_data(
            keep_ignores=kwargs["keep_ignores"],
            keep_rejects=kwargs["keep_rejects"],
            label_reject_and_ignore_as_irrelevant=kwargs["label_reject_and_ignore_as_irrelevant"],
            drop_conflicting_answers=kwargs["drop_conflicting_answers"]
        )
    if use_chatgpt_annotated_data:
        df_chatgpt = load_chatgpt_annotated_data()
    if use_human_annotated_data and use_chatgpt_annotated_data:
        logger.info("Combining human and ChatGPT annotated data to final dataset.")
        logger.info("Checking for label conflicts between human and ChatGPT annotated data.")
        human_unique_labels = df_human["label"].unique()
        chatgpt_unique_labels = df_chatgpt["label"].unique()
        # Use these lines to check if ValueError is raised
        #print(f"Human unique labels:")
        #for label in human_unique_labels:
        #    print(f" {label}")
        #print(f"ChatGPT unique labels:")
        #for label in chatgpt_unique_labels:
        #    print(f" {label}")
        for label in human_unique_labels:
            if label not in chatgpt_unique_labels:
                logger.warning(f"Human label '{label}' not found in ChatGPT annotated data.")
                if label in human_categories_to_chatgpt_categories:
                    logger.info(f"Found custom mapping for human label '{label}' to '{human_categories_to_chatgpt_categories[label]}'.")
                    new_label = human_categories_to_chatgpt_categories[label]
                    logger.info(f"Renaming human label '{label}' to '{new_label}'.")
                    df_human.loc[df_human["label"] == label, "label"] = new_label
                else:
                    raise ValueError(f"Label '{label}' from human annotated data not found in ChatGPT annotated data.")
        df = pd.concat([df_human, df_chatgpt])
        duplicates = df[df.duplicated(subset=["article_id", "section_id"])]
        if not duplicates.empty:
            logger.warning(f"Found {len(duplicates)} duplicates in the final dataset. Dropping duplicates.")
            df.drop(duplicates.index, inplace=True)
    else:
        df = df_human if use_human_annotated_data else df_chatgpt
    if combine_categories:
        logger.info(f"Combining categories into new categories. Number of categories before combining: {len(df.label.unique())}")
        df = merge_categories(df) 
        logger.info(f"Number of categories after combining: {len(df.label.unique())}")
    label_counts = df.label.value_counts()
    logger.info(f"Found {len(label_counts)} unique labels in the dataset with the following counts: {label_counts.to_dict()}")
    low_count_labels = label_counts[label_counts <= min_samples_per_label]
    if not low_count_labels.empty:
        logger.warning(f"Removing labels with less than {min_samples_per_label} samples: {low_count_labels}")
        df = df[~df["label"].isin(low_count_labels.index)]
    if undersample and upsample:
        raise ValueError("Cannot undersample and upsample the data at the same time.")
    if undersample:
        logger.info("Undersampling the data.")
        min_samples = df["label"].value_counts().min()
        df = df.groupby("label", as_index=False).apply(lambda x: x.sample(min_samples))
        logger.info(f"Undersampled data to {len(df)} samples ({min_samples} per label).")
    if upsample:
        logger.info("Upsampling the data.")
        max_samples = df["label"].value_counts().max()
        df = df.groupby("label", as_index=False).apply(lambda x: x.sample(max_samples, replace=True))
        logger.info(f"Upsampled data to {len(df)} samples ({max_samples} per label).")
    if debug:
        dfs = []
        for label in df.label.unique():
            dfs.append(df[df["label"] == label].sample(10))
        df = pd.concat(dfs)
        logger.info(f"Debug mode: Reduced dataset to {len(df)} samples. From each label 10 samples are taken.")
    return df


def get_data_loaders(
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        **kwargs
    ) -> Tuple[Dict[str, DataLoader], preprocessing.LabelEncoder, AutoTokenizer]:
    df = load_data(**kwargs)
    df, le = encode_labels(df, kwargs["checkpoints_dir"]) 
    train_df, val_df, test_df = split_data_into_train_val_test(df, kwargs["train_size"], kwargs["val_size"], kwargs["test_size"])
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    tokenizer = init_tokenizer(train_df, kwargs["model_name"])
    logger.info(f"Size of train_df: {len(train_df)}")
    logger.info(f"Size of val_df: {len(val_df)}")
    logger.info(f"Size of test_df: {len(test_df)}")
    dataloaders = {
        "train": DataLoader(TAndIDataSet(train_df, tokenizer, le), batch_size=train_batch_size, shuffle=True,
                            generator=Generator().manual_seed(2147483647)),
        "val": DataLoader(TAndIDataSet(val_df, tokenizer, le), batch_size=val_batch_size, shuffle=True,
                          generator=Generator().manual_seed(2147483647)),
        "test": DataLoader(TAndIDataSet(test_df, tokenizer, le), batch_size=test_batch_size, shuffle=True,
                           generator=Generator().manual_seed(2147483647))}
    tokenizer.save_pretrained(kwargs['checkpoints_dir'])
    return dataloaders, le, tokenizer
