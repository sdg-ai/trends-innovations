from dataclasses import asdict
import os
from dotenv import load_dotenv

load_dotenv()
import wandb
import torch
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from utils.config import RunConfig, WandbConfig
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from datetime import datetime, timedelta, timezone
from data.datasets import get_data_loaders
from sklearn.metrics import classification_report
from utils.utils import logger as logger, add_file_logger
from utils.metrics import TransformerMetricCollection, AvgDictMeter
from utils.utils import (EarlyStopper, seed_everything, init_configurations,
                         init_wandb)
from transformers import (RobertaForSequenceClassification, get_scheduler,
                           AlbertForSequenceClassification,
                           DistilBertForSequenceClassification)


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--disable_wandb', action='store_true', default=False)
parser.add_argument('--d',
                    type=str,
                    default=str(
                        datetime.strftime(datetime.now(),
                                          format="%Y-%m-%d %H:%M:%S")))
parser.add_argument('--config_name', type=str)
args = parser.parse_args()

MODELS_LIB = {
    "albert-base-v2": AlbertForSequenceClassification,
    "distilbert-base-uncased": DistilBertForSequenceClassification,
    "roberta-base": RobertaForSequenceClassification
}


def log_training_progress_to_console(t_start, steps: int, curr_step: int,
                                     train_results) -> None:
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
    elapsed_time = datetime.fromtimestamp(time() - t_start, timezone.utc)
    log_msg += f" - time: {elapsed_time.strftime('%d-%H:%M:%S')}s"
    time_per_epoch = ((time() - t_start) /
                      curr_step) if curr_step > 0 else time() - t_start
    remaining_time = (steps - curr_step) * time_per_epoch
    time_left = int(remaining_time)
    time_duration = timedelta(seconds=time_left)
    days = time_duration.days
    hours = time_duration.seconds // 3600
    minutes = (time_duration.seconds // 60) % 60
    seconds = time_duration.seconds % 60
    log_msg += f" - remaining time: {days}d-{hours}h-{minutes}m-{seconds}s"
    logger.info(log_msg)


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


def train(model, train_loader: DataLoader, val_loader: DataLoader,
          config: RunConfig, log_dir: str):
    """
    The training loop for the transformer model
    :param model: the model to train
    :param train_loader: the torch dataloader for the training data
    :param val_loader:  the torch dataloader for the validation data
    :param config: the config for the training
    :param log_dir: the directory to save the model to
    """
    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # init lr scheduler
    lr_scheduler = get_scheduler(name="linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=config.num_warmup_steps,
                                 num_training_steps=config.epochs *
                                 len(train_loader))
    early_stopper = EarlyStopper(patience=config.patience)
    avg_train_loss_meter = AvgDictMeter()
    best_val_loss = np.inf
    i_step = 0
    total_steps = config.epochs * (len(train_loader) + len(val_loader))
    t_start = time()
    log_every_i_steps = int(total_steps / config.epochs / 10)
    for epoch in range(config.epochs):
        logger.info(f"\nRunning Epoch {epoch + 1}/{config.epochs}...")
        # training loop
        model.train()
        for batch, _ in train_loader:
            # move batch to gpu
            batch = {k: v.to(config.device) for k, v in batch.items()}
            loss, _ = train_step(model, batch, optimizer)
            avg_train_loss_meter.add({"train_loss": loss})
            lr_scheduler.step()
            i_step += 1
            # log 10 times per epoch
            if i_step % log_every_i_steps == 0:
                train_results = avg_train_loss_meter.compute()
                log_training_progress_to_console(t_start=t_start,
                                                 steps=total_steps,
                                                 curr_step=i_step,
                                                 train_results=train_results)
                wandb.log({
                    f'train/{k}': v
                    for k, v in train_results.items()
                },
                          step=i_step)
                avg_train_loss_meter.reset()
            if config.debug:
                break

        val_loss = validation(model, val_loader, config)
        # save best model
        if val_loss < best_val_loss:
            model.save_pretrained(log_dir)
            best_val_loss = val_loss
            logger.info(f"Saved model to {log_dir}")
        if early_stopper.early_stop(val_loss):
            logger.info("Stopping early")
            break
    return model


def validation(model, val_loader, config: RunConfig) -> float:
    model.eval()
    val_metrics = TransformerMetricCollection(n_classes=config.num_labels,
                                              device=config.device).to(
                                                  config.device)
    avg_val_loss_meter = AvgDictMeter()
    # validation loop
    for batch, _ in val_loader:
        batch = {k: v.to(config.device) for k, v in batch.items()}
        with torch.no_grad():
            loss, predictions = val_step(model, batch)
        val_metrics.update(predictions, batch["labels"])
        avg_val_loss_meter.add({"val_loss": loss})
        if config.debug:
            break
    val_results = avg_val_loss_meter.compute()

    # log metrics
    wandb.log({f'val/{k}': v for k, v in val_metrics.compute().items()})
    wandb.log({f'val/{k}': v for k, v in val_results.items()})
    logger.info("Validation Results:" +
                " - ".join([f'{k}: {v:.4f}' for k, v in val_results.items()]))
    logger.info(
        "Validation Metrics:" +
        " - ".join([f'{k}: {v:.4f}'
                    for k, v in val_metrics.compute().items()]))
    return val_results["val_loss"]


def test(model, test_loader: DataLoader, config: RunConfig, le) -> pd.DataFrame:
    """
    Compute the predictions for the test data
    :param model: the model to use
    :param test_loader: the torch dataloader for the test data
    :param config: the config for the model
    :param le: the label encoder for the test data
    :return: a dataframe containing the predictions from the test data
    """
    predictions = []
    metrics = TransformerMetricCollection(n_classes=config.num_labels,
                                          device=config.device).to(
                                              config.device)
    model.eval()
    for batch, batch_details in test_loader:
        batch = {k: v.to(config.device) for k, v in batch.items()}
        with torch.no_grad():
            _, preds = val_step(model, batch)
            metrics.update(preds, batch["labels"])
            for idx, pred in enumerate(preds.tolist()):
                predictions.append({
                    "y_hat_enc":
                    pred,
                    "y_enc":
                    batch["labels"].flatten().tolist()[idx],
                    "text":
                    batch_details["text"][idx],
                    "article_id":
                    batch_details["article_id"][idx].item(),
                    "section_id":
                    batch_details["section_id"][idx].item()
                })
    predictions = pd.DataFrame(predictions)
    metrics = metrics.compute()
    performance = classification_report(predictions["y_enc"],
                                        predictions["y_hat_enc"],
                                        target_names=le.classes_,
                                        output_dict=True)
    performance = pd.DataFrame(performance).transpose()
    performance["label"] = performance.index.astype(str)
    performance.reset_index(drop=True, inplace=True)
    data = performance.values.tolist()
    columns = performance.columns.tolist()
    wandb_performance_table = wandb.Table(data=data,
                                          columns=columns,
                                          allow_mixed_types=True)
    wandb.log({"test/performance": wandb_performance_table})
    wandb.log({f'test/{k}': v for k, v in metrics.items()})
    logger.info("Test Metrics:" +
                " - ".join([f'{k}: {v:.4f}' for k, v in metrics.items()]))
    with open(
            config.checkpoints_dir +
            f"/seed_{config.seed}/test_metrics.json", "w") as f:
        metrics_to_dump = {k: v.item() for k, v in metrics.items()}
        json.dump(metrics_to_dump, f)
    predictions.to_csv(config.checkpoints_dir +
                       f"/seed_{config.seed}/predictions.csv",
                       index=False)
    return predictions


def run_single_config(run_config: RunConfig,
               wandb_config: WandbConfig,
               sweep=False) -> None:
    init_wandb(args.config_name, run_config, wandb_config, sweep=sweep)
    curr_log_dir = run_config.checkpoints_dir + f"/seed_{run_config.seed}"
    os.makedirs(curr_log_dir, exist_ok=True)
    with open(curr_log_dir + "/run_config.json", "w") as f:
        f.write(str(run_config))
    add_file_logger(curr_log_dir + "/run.log")
    logger.info(f"Running config: {asdict(run_config)}")
    logger.info(f"Loading data.")
    data_loaders, le, tokenizer = get_data_loaders(config=run_config, debug=args.debug)
    run_config.num_labels = le.classes_.shape[0]
    # run seeds
    wandb.run.summary["train_size"] = len(data_loaders["train"].dataset)
    seed_everything(run_config.seed)
    # init model
    current_model = MODELS_LIB[run_config.model_name].from_pretrained(
        run_config.model_name, num_labels=len(le.classes_)).to(run_config.device)
    # resize token embeddings
    current_model.resize_token_embeddings(len(tokenizer))
    if not sweep:
        init_wandb(args.config_name, run_config, wandb_config, data_loaders)
    logger.info("----- TRAINING -----")
    final_model = train(current_model, data_loaders["train"],
                        data_loaders["val"], run_config, curr_log_dir)
    logger.info("----- TESTING -----")
    test(final_model, data_loaders["test"], run_config, le)
    wandb.finish()


def run_single_config_as_sweep(run_config: RunConfig, wandb_config: WandbConfig) -> None:
    """
    Runs a sweep configuration. Simply excutes the run_config function for each sweep configuration.
    :param config_name: the name of the configuration to run
    """
    sweep_id = wandb.sweep(run_config.sweep_config, project=wandb_config.project)
    sweep_func = lambda: run_config(args.config_name, run_config, wandb_config, sweep=True)
    wandb.agent(sweep_id, function=sweep_func)


def run_all_configs(configurations: Dict[str, Tuple[RunConfig, WandbConfig]]) -> None:
    """
    Runs all configurations. Simply excutes the run_config function for each configuration and each seed.
    """
    for config_name, (run_config, wandb_config) in configurations.items():
        logger.info(f"Using device: {run_config.device}")
        logger.warning("CUDA not available" if run_config.device ==
                       "cpu" else "CUDA available")
        for seed in range(run_config.num_seeds):
            logger.info(f"-------- RUNNING SEED {seed} --------")
            run_config.seed = run_config.initial_seed + seed
            run_single_config(run_config, wandb_config)


if __name__ == "__main__":
    configurations = init_configurations(args)
    if args.config_name and configurations[args.config_name].sweep_config:
        run_single_config_as_sweep(**configurations[args.config_name])
    elif args.config_name:
        run_single_config(**configurations[args.config_name])
    else:
        run_all_configs(configurations)
