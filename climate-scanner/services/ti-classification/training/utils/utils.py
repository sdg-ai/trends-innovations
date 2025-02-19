from dataclasses import asdict
import os
import yaml
import wandb
import logging
import torch
import random
import numpy as np
from typing import Dict, Tuple
from utils.config import RunConfig, WandbConfig

WANDB_KEY = os.environ.get("WANDB_KEY") or ""
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or ""
wandb.login(key=WANDB_KEY)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


class EarlyStopper:
    """
    Stops the training early if there is no improvement in the validation loss for a given number of epochs
    :param patience: the number of epochs to wait for the validation loss to improve
    :param min_delta: the minimum change in the validation loss to be considered as an improvement
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        """
        Checks if the training should be stopped early.
        :param validation_loss: the current validation loss
        :return: True if validation loss is greater than the minimum validation loss for longer than the patience
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def seed_everything(seed):
    """
    Sets the seed for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_configurations(args) -> Dict[str, Tuple[RunConfig, Dict]]:
    with open("train_configs.yml", "r") as f:
        custom_configs = yaml.safe_load(f)
    initialized_configs = {}
    
    for cust_config_name, cust_config in custom_configs.items():
        # Create a new DefaultConfig instance instead of using dict
        run_config = RunConfig(debug=args.debug)
       
        # Update with the YAML config values
        for key, value in cust_config.items():
            if hasattr(run_config, key):
                setattr(run_config, key, value)
        
        # Update checkpoints directory
        run_config.checkpoints_dir = f"{run_config.checkpoints_dir}/{args.d}-{cust_config['model_name']}-{cust_config_name}"
        
        # Set seed and model name
        run_config.seed = run_config.initial_seed
        run_config.model_name = cust_config["model_name"]
        
        # Handle wandb config separately
        wandb_dict = cust_config.pop("wandb", {})
        wandb_config = WandbConfig(disable_wandb=args.disable_wandb)
        for key, value in wandb_dict.items():
            if hasattr(wandb_config, key):
                setattr(wandb_config, key, value)
        
        if cust_config.get("skip", False):
            continue
            
        initialized_configs[cust_config_name] = (run_config, wandb_config)
    return initialized_configs


def init_wandb(config_name, run_config:RunConfig, wandb_config:WandbConfig, sweep=False) -> None:
    wandb.init(
        entity=WANDB_ENTITY,
        project=wandb_config.project,
        config=asdict(run_config),
        mode="disabled" if wandb_config.disabled else "online",
        group=
        f"{run_config.data_dir}-{config_name}{('-' + wandb_config.group_name_modifier) if wandb_config.group_name_modifier != '' else ''}",
        job_type=f"train-{wandb_config.job_type_modifier}"
        if wandb_config.job_type_modifier != '' else "train",
        name="seed_" + str(run_config.seed) if not sweep else None,
        tags=["debug" if run_config.debug else "valid", run_config.model_name],
    )


def add_file_logger(log_path) -> None:
    """
    Adds a file logger to the specified path, removing all previous file loggers.
    :param log_path: the path to the log file
    """
    # Remove all previous file loggers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Add file logger
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
