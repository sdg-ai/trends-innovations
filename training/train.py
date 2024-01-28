import os
import yaml
import wandb
import torch
import logging
# set logging level to info
logging.basicConfig(level=logging.INFO)
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from typing import Tuple, Dict
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from data.datasets import get_data_loaders, get_data_loaders_with_generated_data, get_data_loaders_with_chatgpt_annotated_data
from utils.utils import EarlyStopper, seed_everything
from utils.metrics import TransformerMetricCollection, AvgDictMeter
from transformers import RobertaForSequenceClassification, get_scheduler, AlbertForSequenceClassification, DistilBertForSequenceClassification

load_dotenv()

WANDB_KEY = os.environ.get("WANDB_KEY") or ""
wandb.login(key=WANDB_KEY)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--disable_wandb', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='old_data')
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S")))
args = parser.parse_args()


WANDB_CONFIG = {
    "entity": "j-getzner",
    "project": "Trends & Innovations Classifier",
    "disabled": True,
    "job_type_modifier": ""
}

DEFAULT_CONFIG = {
    # data details
    "data_dir": "./datasets",
    "num_labels": 17 if (args.dataset == "old_data" or args.dataset == "generated_data") else 57,
    "dataset_splits": [0.7, 0.9],

    # model details
    "model_name": "distilbert-base-uncased",
    "lr": 5e-5,
    "epochs": 25 if not args.debug else 5,
    "patience": 5,
    "batch_sizes": {
        "train": 64,
        "val": 64,
        "test": 64
    } if not args.debug else {
        "train": 3,
        "val": 3,
        "test": 3
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
    logging.info(log_msg)


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
    log_every_i_steps = int(total_steps/config["epochs"]/10)
    for epoch in range(config["epochs"]):
        logging.info(f"\nRunning Epoch {epoch + 1}/{config['epochs']}...")
        # training loop
        model.train()
        for batch in train_loader:
            # move batch to gpu
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            loss, predictions = train_step(model, batch, optimizer)
            avg_train_loss_meter.add({"train_loss": loss})
            lr_scheduler.step()
            i_step += 1

            # log 10 times per epoch

            if i_step % log_every_i_steps == 0:
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
            logging.info(f"Saved model to {log_dir}")
        if early_stopper.early_stop(val_loss):
            logging.info("Stopping early")
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
    logging.info("Validation Results:" + " - ".join([f'{k}: {v:.4f}' for k, v in val_results.items()]))
    logging.info("Validation Metrics:" + " - ".join([f'{k}: {v:.4f}' for k, v in val_metrics.compute().items()]))
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


def test(model, test_loader: DataLoader, config: Dict, le) -> pd.DataFrame:
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

    predictions = pd.DataFrame(predictions)
    metrics = metrics.compute()
    wandb.log({"test/conf_mat": wandb.plot.confusion_matrix(
        preds=predictions["y_hat_enc"].tolist(),
        y_true=predictions["y_enc"].tolist(),
        class_names=le.inverse_transform(range(config["num_labels"]))
    )})
    wandb.log({f'test/{k}': v for k, v in metrics.items()})
    logging.info("Test Metrics:" + " - ".join([f'{k}: {v:.4f}' for k, v in metrics.items()]))
    return predictions


def init_configurations():
    with open("./train_run_configs.yml", "r") as f:
        custom_configs = yaml.safe_load(f)
    initialized_configs = []
    for config_name, config in custom_configs.items():
        run_config = DEFAULT_CONFIG.copy()
        run_config.update(config)
        run_config["save_model_dir"] = f"{run_config['save_model_dir']}/{args.d}-{args.model_name}"
        run_config["seed"] = run_config["initial_seed"]
        run_config["model_name"] = args.model_name
        wandb_config = WANDB_CONFIG.copy()
        wandb_config.update(config["wandb"])
        initialized_configs.append((run_config, wandb_config))
    return initialized_configs


if __name__ == "__main__":
    configs = init_configurations()
    for current_config, current_wandb_config in configs:
        # load data
        logging.info(f"Running config: {current_config}")
        logging.info(f"Loading data.")
        if args.dataset == "old_data":
            data_loading_func = get_data_loaders
        elif args.dataset == "generated_data":
            data_loading_func = get_data_loaders_with_generated_data
        elif args.dataset == "chatgpt_annotated_data":
            data_loading_func = get_data_loaders_with_chatgpt_annotated_data
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        data_loaders, le, tokenizer = data_loading_func(current_config, debug=args.debug)
        for seed in range(current_config["num_seeds"]):
            # seed
            logging.info(f"-------- RUNNING SEED {seed} --------")
            current_config["seed"] = current_config["initial_seed"] + seed
            seed_everything(current_config["seed"])
            # change save model dir
            curr_log_dir = current_config["save_model_dir"] + f"/seed_{current_config['seed']}"
            # init model
            current_model = TRANSFORMERS_LIB[current_config["model_name"]].from_pretrained(
                current_config["model_name"],
                num_labels=current_config["num_labels"]
            ).to(current_config["device"])
            current_model.resize_token_embeddings(len(tokenizer))
            wandb.init(
                entity=current_wandb_config["entity"],
                project=current_wandb_config["project"],
                config=current_config,
                mode="disabled" if args.disable_wandb else "online",
                group=f"{args.d}-{current_config['model_name']}",
                job_type="train" + current_wandb_config["job_type_modifier"],
                name="seed_"+str(current_config["seed"]),

            )
            wandb.run.summary["train_size"] = len(data_loaders["train"].dataset)
            if args.dataset == "generated_data":
                df = data_loaders["train"].dataset.data
                wandb.run.summary["generated_data_size"] = len(df.loc[df.generated == True])
                wandb.run.summary["generated_article_labels"] = df.loc[df.generated == True].label.unique()
            wandb.run.summary["val_size"] = len(data_loaders["val"].dataset)
            wandb.run.summary["test_size"] = len(data_loaders["test"].dataset)
            logging.info("----- TRAINING -----")
            final_model = train(current_model, data_loaders["train"], data_loaders["val"], current_config, curr_log_dir)
            logging.info("----- TESTING -----")
            test(final_model, data_loaders["test"], current_config, le)
            wandb.finish()
