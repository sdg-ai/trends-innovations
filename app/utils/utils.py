import numpy as np
import random
import torch
import os
from sklearn.model_selection import train_test_split

SEED = 42


def split_dataset(dataset, val_size=0.2, test_size=0.1):
    """
    Splits the dataset into training, validation and test sets.
    """
    # Split the dataset into training and test sets
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=SEED)
    # Split the training set into training and validation sets
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=SEED)
    return train_data, val_data, test_data


def seed_everything():
    """
    Sets the seed for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


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
