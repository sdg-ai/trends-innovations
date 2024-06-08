import os
import yaml
import wandb
import logging
import torch
import random
import numpy as np

WANDB_KEY = os.environ.get("WANDB_KEY") or ""
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or ""

wandb.login(key=WANDB_KEY)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logger's level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

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


def init_configurations(args, DEFAULT_CONFIG, WANDB_CONFIG):
    with open("training/train_configs.yml", "r") as f:
        custom_configs = yaml.safe_load(f)
    initialized_configs = {}
    for config_name, config in custom_configs.items():
        run_config = DEFAULT_CONFIG.copy()
        run_config.update(config)
        # directory where checkpoints are saved is modified to include the date and model name
        run_config["checkpoints_dir"] = f"{run_config['checkpoints_dir']}/{args.d}-{config['model_name']}"
        # set seed
        run_config["seed"] = run_config["initial_seed"]
        # set model name
        run_config["model_name"] = config["model_name"]
        # create seperate config dict for wandb
        run_config.pop("wandb")
        wandb_config = WANDB_CONFIG.copy()
        wandb_config.update(config["wandb"])
        # add args to config
        run_config.update(vars(args))
        initialized_configs[config_name] = (run_config, wandb_config)
    return initialized_configs


def init_wandb(config_name, config, wandb_config, data_loaders):
    wandb.init(
        entity=WANDB_ENTITY,
        project=wandb_config["project"],
        config=config,
        mode="disabled" if wandb_config["disabled"] else "online",
        group=f"{config['d']}-{config_name}{('-' + wandb_config['group_name_modifier']) if wandb_config['group_name_modifier'] != '' else ''}",
        job_type=f"train-{wandb_config['job_type_modifier']}" if wandb_config["job_type_modifier"] != '' else "train",
        name="seed_"+str(config["seed"]),
        tags=["debug" if config["debug"] else "valid", config["model_name"]],
    )
    wandb.run.summary["train_size"] = len(data_loaders["train"].dataset)
    if config["dataset"] == "generated_data":
        df = data_loaders["train"].dataset.data
        wandb.run.summary["generated_data_size"] = len(df.loc[df.generated == True])
        wandb.run.summary["generated_article_labels"] = df.loc[df.generated == True].label.unique()
        wandb.run.summary["val_size"] = len(data_loaders["val"].dataset)
        wandb.run.summary["test_size"] = len(data_loaders["test"].dataset)
    return wandb.config


def add_file_logger(log_path):
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