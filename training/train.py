import argparse
from dotenv import load_dotenv
import wandb
import torch
import os
from datetime import datetime
from typing import Tuple
from torch.utils.data import DataLoader
from data.datasets import get_data_loaders
from utils.utils import EarlyStopper, seed_everything
from utils.metrics import TransformerMetricCollection, AvgDictMeter
from transformers import RobertaForSequenceClassification, get_scheduler, \
    AlbertForSequenceClassification, DistilBertForSequenceClassification
from tqdm import tqdm
import numpy as np
import pandas as pd
load_dotenv()

WANDB_KEY = os.environ.get("WANDB_KEY") or ""
print("hello")
print(WANDB_KEY)
wandb.login(key=WANDB_KEY)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
args = parser.parse_args()


WANDB_CONFIG = {
  "entity": "j-getzner",
  "project": "Trends & Innovations Classifier",
  "disabled": False
}

DEFAULT_CONFIG = {
    # data details
    "data_dir": "./datasets/annotated_data",
    "num_labels": 17,
    "dataset_splits": [0.7, 0.9],

    # model details
    "model_name": "distilbert-base-uncased",
    "lr": 1e-5,
    "epochs": 20,
    "patience": 3,
    "batch_sizes": {
        "train": 5,
        "val": 5,
        "test": 5
    },
    # other details
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "initial_seed": 1,
    "num_seeds": 1,
    "save_model_dir": "./checkpoints",
}

TRANSFORMERS_LIB = {
    "albert-base-v2": AlbertForSequenceClassification,
    "distilbert-base-uncased": DistilBertForSequenceClassification,
    "roberta-base": RobertaForSequenceClassification
}


def train(model, train_loader: DataLoader, val_loader: DataLoader, config):
    """
    The training loop for the transformer model
    :param model: the model to train
    :param train_loader: the torch dataloader for the training data
    :param val_loader:  the torch dataloader for the validation data
    :param config: the config for the training
    """
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    # init lr scheduler
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=config["epochs"] * len(train_loader))
    # init progress bar
    progress_bar = tqdm(range(config["epochs"] * (len(train_loader) + len(val_loader))))
    # define metrics
    early_stopper = EarlyStopper(patience=config["patience"])
    train_metrics = TransformerMetricCollection(n_classes=config["num_labels"], device=config["device"]).to(config["device"])
    val_metrics = TransformerMetricCollection(n_classes=config["num_labels"], device=config["device"]).to(config["device"])
    avg_loss_meter = AvgDictMeter()
    # keep track of best val loss
    best_val_loss = np.inf
    for epoch in range(config["epochs"]):
        print(f"\nRunning Epoch {epoch + 1}/{config['epochs']}...")
        # training loop
        model.train()
        for batch in train_loader:
            # move batch to gpu
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            loss, predictions = train_step(model, batch, optimizer)
            train_metrics.update(predictions, batch["labels"])
            avg_loss_meter.add({"train_loss": loss})
            lr_scheduler.step()
            progress_bar.update(1)

        # validation loop
        model.eval()
        for batch in val_loader:
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            with torch.no_grad():
                loss, predictions = val_step(model, batch)
            val_metrics.update(predictions, batch["labels"])
            avg_loss_meter.add({"val_loss": loss})
            progress_bar.update(1)

        # compute avg loss
        mean_epoch_loss = avg_loss_meter.compute()

        # log metrics
        wandb.log({f'train/{k}': v for k, v in train_metrics.compute().items()})
        wandb.log({f'val/{k}': v for k, v in val_metrics.compute().items()})
        wandb.log({f'loss/{k}': v for k, v in mean_epoch_loss.items()})
        train_metrics.reset()
        val_metrics.reset()

        # save best model
        if mean_epoch_loss["val_loss"] < best_val_loss:
            model_path = f"{config['save_model_dir']}"
            model.save_pretrained(model_path)
            best_val_loss = mean_epoch_loss["val_loss"]
        if early_stopper.early_stop(mean_epoch_loss["val_loss"]):
            print("Stopping early")
            break


def train_step(model, batch: dict, optimizer) -> Tuple[float, torch.Tensor]:
    """
    Performs a single training step with backpropagation
    :param model: the model to use
    :param optimizer: the optimizer to use
    :param batch: the batch to train on
    :return: the loss for the current batch and the predictions
    """
    # forward pass
    output = model(**batch)
    predictions = torch.argmax(output.logits, dim=-1)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(), predictions


def val_step(model, batch: dict) -> Tuple[float, torch.Tensor]:
    """
    Performs a single validation step
    :param model: the model to use
    :param batch: the batch to train on
    :return: the loss for the current batch and the predictions
    """
    with torch.no_grad():
        output = model(**batch)
        predictions = torch.argmax(output.logits, dim=-1)
    return output.loss.item(), predictions


def test(model, test_loader: DataLoader, config) -> pd.DataFrame:
    """
    Compute the predictions for the test data
    :param model: the model to use
    :param test_loader: the torch dataloader for the test data
    :param config: the config for the model
    :return: a dataframe containing the predictions from the test data
    """
    progress_bar = tqdm(range(len(test_loader)))
    predictions = []
    metrics = TransformerMetricCollection(n_classes=config["num_labels"], device=config["device"]).to(config["device"])
    model.eval()
    for batch in test_loader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        with torch.no_grad():
            _, preds = val_step(model, batch)
            metrics.update(preds, batch["labels"])
            for idx, pred in enumerate(preds.tolist()):
                predictions.append({"y_hat_enc": pred, "y_enc": batch["labels"].flatten().tolist()[idx], })
        progress_bar.update(1)
    metrics = metrics.compute()
    print(metrics)
    wandb.log({"test": metrics})
    predictions = pd.DataFrame(predictions)
    return predictions


if __name__ == "__main__":
    current_time = datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")
    current_config = DEFAULT_CONFIG.copy()
    current_config["seed"] = current_config["initial_seed"]
    if args.model_name:
        current_config["model_name"] = args.model_name
    for seed in range(current_config["num_seeds"]):
        # seed
        current_config["seed"] = current_config["initial_seed"] + seed
        seed_everything(current_config["seed"])
        # change save model dir
        current_config["save_model_dir"] = f"{current_config['save_model_dir']}/{current_config['model_name']}/seed_{current_config['seed']}"
        # init model
        current_model = TRANSFORMERS_LIB[current_config["model_name"]].from_pretrained(
            current_config["model_name"],
            num_labels=current_config["num_labels"]
        ).to(current_config["device"])

        # load data
        data_loaders = get_data_loaders(current_config)

        wandb.init(
            entity=WANDB_CONFIG["entity"],
            project=WANDB_CONFIG["project"],
            config=current_config,
            mode="disabled" if WANDB_CONFIG["disabled"] else "online",
            group=f"{current_time}-{current_config['model_name']}",
            job_type="train",
            name="seed_"+str(current_config["seed"])
        )

        train(current_model, data_loaders["train"], data_loaders["val"], current_config)
        test(current_model, data_loaders["test"], current_config)
