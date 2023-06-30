import numpy as np
import random
import torch
import os


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
