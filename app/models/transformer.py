import os
from typing import Tuple

import torch
import wandb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from ._model import TandIClassifier
from ..utils import EarlyStopper
from ..utils.metrics import TransformerMetricCollection, AvgDictMeter
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_scheduler, \
    AlbertForSequenceClassification, DistilBertForSequenceClassification, BatchEncoding

transformers_lib = {
    "albert-base-v2": AlbertForSequenceClassification,
    "distilbert-base-uncased": DistilBertForSequenceClassification,
    "roberta-base": RobertaForSequenceClassification
}


class TransformerTandIClassifier(TandIClassifier):
    """
    Wrapper class for hugging face transformer models
    model_name: the name of the hugger face model (must be in the transformers_lib dict)
    model_config: the model config dict
    save_model_dir: the directory to save the model after training
    wandb: dictionary with wandb config dict
    TODO: custom tokens/vocabulary
    custom_tokens: TBD
    """
    def __init__(self, model_name: str, model_config: dict, save_model_dir="/checkpoints", wandb_config=None, custom_tokens=None):
        self.model_name = model_name
        self.save_model_dir = save_model_dir
        self.wandb = wandb_config
        self.model_config = model_config
        # check if there is already a saved model
        # TODO: make path configurable
        if os.path.exists(f"app/checkpoints/{model_name}"):
            print("Loading preexisting model...")
            self.model = transformers_lib[model_name].from_pretrained(
                f"app/checkpoints/{model_name}",
                local_files_only=True
            ).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(f"app/checkpoints/{model_name}", local_files_only=True)
        else:
            self.model = transformers_lib[model_name].from_pretrained(
                model_name,
                num_labels=model_config["num_labels"]
            ).to("cuda")
            # initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # add custom tokens to the tokenizer if specified
            if custom_tokens:
                new_tokens = set(custom_tokens) - set(self.tokenizer.vocab.keys())
                self.tokenizer.add_tokens(list(new_tokens))
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=model_config["lr"])

    def predict(self, samples: BatchEncoding):
        """
        Predicts the labels for the given samples
        :param samples: the samples to predict
        :return: the predictions
        """
        samples = samples.to("cuda")
        self.model.eval()
        with torch.no_grad():
            output = self.model(**samples, output_attentions=False, output_hidden_states=False)
        predictions = torch.argmax(output.logits, dim=-1)
        return predictions

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        the training loop for the transformer model
        :param train_loader: the torch dataloader for the training data
        :param val_loader:  the torch dataloader for the validation data
        """
        # init lr scheduler
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.model_config["epochs"] * len(train_loader)
        )
        # init progress bar
        progress_bar = tqdm(range(self.model_config["epochs"] * (
                train_loader.batch_size * len(train_loader) + val_loader.batch_size * len(val_loader))))
        # define metrics
        early_stopper = EarlyStopper(patience=self.model_config["patience"])
        train_metrics = TransformerMetricCollection(n_classes=self.model_config["num_labels"]).to("cuda")
        val_metrics = TransformerMetricCollection(n_classes=self.model_config["num_labels"]).to("cuda")
        avg_loss_meter = AvgDictMeter()
        # keep track of best val loss
        best_val_loss = np.inf
        # init weights and biases
        wandb.init(
            entity=self.wandb["entity"],
            project=self.wandb["project"],
            config=self.model_config,
            mode="disabled" if self.wandb["disabled"] else "online"
        )
        for epoch in range(self.model_config["epochs"]):
            print(f"\nRunning Epoch {epoch + 1}/{self.model_config['epochs']}...")
            # training loop
            self.model.train()
            for batch in train_loader:
                # move batch to gpu
                batch = {k: v.to("cuda") for k, v in batch.items()}
                loss, predictions = self.train_step(batch)
                train_metrics.update(predictions, batch["labels"])
                avg_loss_meter.add({"train_loss": loss})
                lr_scheduler.step()
                progress_bar.update(1)

            # validation loop
            self.model.eval()
            for batch in val_loader:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.no_grad():
                    loss, predictions = self.val_step(batch)
                val_metrics.update(predictions, batch["labels"])
                avg_loss_meter.add({"val_loss": loss})
                progress_bar.update(1)

            # compute avg loss
            mean_epoch_loss = avg_loss_meter.compute()

            # log metrics
            wandb.log({"train": train_metrics.compute()}, step=epoch)
            wandb.log({"val": val_metrics.compute()}, step=epoch)
            wandb.log(mean_epoch_loss, step=epoch)
            train_metrics.reset()
            val_metrics.reset()

            # save best model
            if mean_epoch_loss["val_loss"] < best_val_loss:
                model_path = f"{self.save_model_dir}/{self.model_name}"
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                best_val_loss = mean_epoch_loss["val_loss"]
            if early_stopper.early_stop(mean_epoch_loss["val_loss"]):
                print("Stopping early")
                break

    def train_step(self, batch:dict) -> Tuple[float, torch.Tensor]:
        """
        Performs a single training step with backpropagation
        :param batch: the batch to train on
        :return: the loss for the current batch and the predictions
        """
        # forward pass
        output = self.model(**batch)
        predictions = torch.argmax(output.logits, dim=-1)
        loss = output.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), predictions

    def val_step(self, batch:dict) -> Tuple[float, torch.Tensor]:
        """
        Performs a single validation step
        :param batch: the batch to train on
        :return: the loss for the current batch and the predictions
        """
        with torch.no_grad():
            output = self.model(**batch)
            predictions = torch.argmax(output.logits, dim=-1)
        return output.loss.item(), predictions

    def test(self, test_loader: DataLoader) -> pd.DataFrame:
        """
        Compute the predictions for the test data
        :param test_loader: the torch dataloader for the test data
        :return: a dataframe containing the predictions
        """
        progress_bar = tqdm(range(len(test_loader)))
        predictions = []
        metrics = TransformerMetricCollection(n_classes=self.model_config["num_labels"]).to("cuda")
        self.model.eval()
        for batch in test_loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                _, preds = self.val_step(batch)
                metrics.update(preds, batch["labels"])
                for idx, pred in enumerate(preds.tolist()):
                    predictions.append({"y_hat_enc": pred, "y_enc": batch["labels"].flatten().tolist()[idx], })
            progress_bar.update(1)
        metrics = metrics.compute()
        print(metrics)
        self.wandb.log({"test": metrics})
        predictions = pd.DataFrame(predictions)
        return predictions
