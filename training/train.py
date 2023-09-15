import os
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from typing import Tuple, Dict
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from data.datasets import get_data_loaders
from utils.utils import EarlyStopper, seed_everything
from utils.metrics import TransformerMetricCollection, AvgDictMeter
from transformers import RobertaForSequenceClassification, get_scheduler, AlbertForSequenceClassification, DistilBertForSequenceClassification

load_dotenv()

WANDB_KEY = os.environ.get("WANDB_KEY") or ""
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
    "lr": 5e-5,
    "epochs": 25,
    "patience": 5,
    "batch_sizes": {
        "train": 16,
        "val": 64,
        "test": 64
    },
    # other details
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "initial_seed": 1,
    "num_seeds": 3,
    "save_model_dir": "./checkpoints",
}
print("device:", DEFAULT_CONFIG["device"])
TRANSFORMERS_LIB = {
    "albert-base-v2": AlbertForSequenceClassification,
    "distilbert-base-uncased": DistilBertForSequenceClassification,
    "roberta-base": RobertaForSequenceClassification
}


def log_training_progress_to_console(t_start, steps: int, curr_step: int, train_results) -> None:
    """
    Logs the training progress to the console
    :param t_start:  the start time of the training
    :param steps:  the total number of steps
    :param curr_step:  the current step
    :param train_results:  the training results (loss)
    :return:
    """
    log_msg = " - ".join([f'{k}: {v:.4f}' for k, v in train_results.items()])
    log_msg = f"Iteration {curr_step} - " + log_msg
    elapsed_time = datetime.utcfromtimestamp(time() - t_start)
    log_msg += f" - time: {elapsed_time.strftime('%d-%H:%M:%S')}s"
    time_per_epoch = ((time() - t_start) / curr_step) if curr_step > 0 else time() - t_start
    remaining_time = (steps - curr_step) * time_per_epoch
    time_left = int(remaining_time)
    time_duration = timedelta(seconds=time_left)
    days = time_duration.days
    hours = time_duration.seconds // 3600
    minutes = (time_duration.seconds // 60) % 60
    seconds = time_duration.seconds % 60
    log_msg += f" - remaining time: {days}d-{hours}h-{minutes}m-{seconds}s"
    print(log_msg)


def train(model, train_loader: DataLoader, val_loader: DataLoader, config: Dict, log_dir: str):
    """
    The training loop for the transformer model
    :param model: the model to train
    :param train_loader: the torch dataloader for the training data
    :param val_loader:  the torch dataloader for the validation data
    :param config: the config for the training
    :param log_dir: the directory to save the model to
    """
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    # init lr scheduler
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=config["epochs"] * len(train_loader))
    early_stopper = EarlyStopper(patience=config["patience"])
    avg_train_loss_meter = AvgDictMeter()
    best_val_loss = np.inf
    i_step = 0
    total_steps = config["epochs"] * (len(train_loader) + len(val_loader))
    t_start = time()
    for epoch in range(config["epochs"]):
        print(f"\nRunning Epoch {epoch + 1}/{config['epochs']}...")
        # training loop
        model.train()
        for batch in train_loader:
            # move batch to gpu
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            loss, predictions = train_step(model, batch, optimizer)
            avg_train_loss_meter.add({"train_loss": loss})
            lr_scheduler.step()
            i_step += 1

            if i_step % 100 == 0:
                train_results = avg_train_loss_meter.compute()
                log_training_progress_to_console(
                    t_start=t_start,
                    steps=total_steps,
                    curr_step=i_step,
                    train_results=train_results
                )
                wandb.log({f'train/{k}': v for k, v in train_results.items()}, step=i_step)
                avg_train_loss_meter.reset()

        val_loss = validation(model, val_loader, config)

        # save best model
        if val_loss < best_val_loss:
            model.save_pretrained(log_dir)
            best_val_loss = val_loss
            print(f"Saved model to {log_dir}")
        if early_stopper.early_stop(val_loss):
            print("Stopping early")
            break
    return model


def validation(model, val_loader, config: Dict) -> float:
    model.eval()
    val_metrics = TransformerMetricCollection(
        n_classes=config["num_labels"],
        device=config["device"]
    ).to(config["device"])
    avg_val_loss_meter = AvgDictMeter()
    # validation loop
    for batch in val_loader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        with torch.no_grad():
            loss, predictions = val_step(model, batch)
        val_metrics.update(predictions, batch["labels"])
        avg_val_loss_meter.add({"val_loss": loss})
    val_results = avg_val_loss_meter.compute()

    # log metrics
    wandb.log({f'val/{k}': v for k, v in val_metrics.compute().items()})
    wandb.log({f'val/{k}': v for k, v in val_results.items()})
    print("Validation Results:" + " - ".join([f'{k}: {v:.4f}' for k, v in val_results.items()]))
    print("Validation Metrics:" + " - ".join([f'{k}: {v:.4f}' for k, v in val_metrics.compute().items()]))
    return val_results["val_loss"]


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
    metrics = metrics.compute()
    wandb.log({f'test/{k}': v for k, v in metrics.items()})
    print("Test Metrics:" + " - ".join([f'{k}: {v:.4f}' for k, v in metrics.items()]))
    predictions = pd.DataFrame(predictions)
    return predictions


if __name__ == "__main__":
    current_time = datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S")
    current_config = DEFAULT_CONFIG.copy()
    current_config["save_model_dir"] = f"{current_config['save_model_dir']}/{current_time}-{current_config['model_name']}"
    current_config["seed"] = current_config["initial_seed"]
    if args.model_name:
        current_config["model_name"] = args.model_name
    # load data
    data_loaders = get_data_loaders(current_config)
    for seed in range(current_config["num_seeds"]):
        # seed
        current_config["seed"] = current_config["initial_seed"] + seed
        seed_everything(current_config["seed"])
        # change save model dir
        curr_log_dir = current_config["save_model_dir"] + f"/seed_{current_config['seed']}"
        # init model
        current_model = TRANSFORMERS_LIB[current_config["model_name"]].from_pretrained(
            current_config["model_name"],
            num_labels=current_config["num_labels"]
        ).to(current_config["device"])

        wandb.init(
            entity=WANDB_CONFIG["entity"],
            project=WANDB_CONFIG["project"],
            config=current_config,
            mode="disabled" if WANDB_CONFIG["disabled"] else "online",
            group=f"{current_time}-{current_config['model_name']}",
            job_type="train",
            name="seed_"+str(current_config["seed"])
        )

        final_model = train(current_model, data_loaders["train"], data_loaders["val"], current_config, curr_log_dir)
        print("Testing...")
        test(final_model, data_loaders["test"], current_config)

        wandb.finish()
