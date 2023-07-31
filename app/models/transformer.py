import os
from typing import Tuple

import torch
import wandb
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from ._model import TandIClassifier
from typing import List
from pickle import load

from utils import EarlyStopper, seed_everything
from utils.metrics import TransformerMetricCollection, AvgDictMeter
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_scheduler, \
    AlbertForSequenceClassification, DistilBertForSequenceClassification

transformers_lib = {
    "albert-base-v2": AlbertForSequenceClassification,
    "distilbert-base-uncased": DistilBertForSequenceClassification,
    "roberta-base": RobertaForSequenceClassification
}


class TransformerTandIClassifier(TandIClassifier):
    """
    Wrapper class for fine-tuning hugging face BERT transformer models for the task of text classification
    """
    def __init__(self, model_name: str, model_config: dict, save_model_dir="/checkpoints", custom_tokens=None):
        """
        :param model_name: the name of the huggingface model (as specified in transformers_lib)
        :param model_config: the model config dict (see train_config.yml)
        :param save_model_dir: the directory to save the model to or load the model from
        :param custom_tokens: a list of words, that should be added to the tokenizer
        """
        self.model_name = model_name
        self.save_model_dir = save_model_dir
        self.config = model_config
        self.model, self.tokenizer = self.load_model()
        # add custom tokens to the tokenizer if specified
        if custom_tokens:
            new_tokens = set(custom_tokens) - set(self.tokenizer.vocab.keys())
            self.tokenizer.add_tokens(list(new_tokens))

    def load_model(self):
        """
        Loads the model from a previous checkpoint if available, otherwise initializes a new model and tokenizer
        :return: the model, the corresponding tokenizer
        """
        # TODO: make path configurable
        # check if there is already a saved model
        if os.path.exists(f"app/checkpoints/{self.model_name}"):
            logging.info("Loading preexisting model...")
            model = transformers_lib[self.model_name].from_pretrained(
                f"app/checkpoints/{self.model_name}",
                local_files_only=True
            ).to(self.config["device"])
            tokenizer = AutoTokenizer.from_pretrained(f"app/checkpoints/{self.model_name}", local_files_only=True)
            logging.info("Model loaded.")
        else:
            print("No previous model checkpoint found.")
            model = transformers_lib[self.model_name].from_pretrained(
                self.model_name,
                num_labels=self.config["num_labels"]).to(self.config["device"])
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def predict(self, samples: List[str]):
        """
        Takes text samples as input, transforms them and predicts the labels for the given samples
        :param samples: the samples to predict
        :return: the predictions
        """

        encodings = self.tokenizer(
            samples,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        encodings = encodings.to(self.config["device"])
        self.model.eval()
        with torch.no_grad():
            output = self.model(**encodings, output_attentions=False, output_hidden_states=False)
        # get softmax of logits to get probabilities
        probs = torch.nn.Softmax()(output.logits)
        probs = torch.max(probs, dim=-1)
        predictions = torch.argmax(output.logits, dim=-1)
        # get values from probs by indexes from predictions

        le = load(open(f'app/checkpoints/{self.model_name}/label_encoder.pkl', 'rb'))
        return list(le.inverse_transform(predictions.tolist())), probs

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        The training loop for the transformer model
        :param train_loader: the torch dataloader for the training data
        :param val_loader:  the torch dataloader for the validation data
        """
        seed_everything(self.config["seed"])
        # define optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])
        # init lr scheduler
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                     num_training_steps=self.config["epochs"] * len(train_loader))
        # init progress bar
        progress_bar = tqdm(range(self.config["epochs"] * (len(train_loader) + len(val_loader))))
        # define metrics
        early_stopper = EarlyStopper(patience=self.config["patience"])
        train_metrics = TransformerMetricCollection(n_classes=self.config["num_labels"]).to(self.config["device"])
        val_metrics = TransformerMetricCollection(n_classes=self.config["num_labels"]).to(self.config["device"])
        avg_loss_meter = AvgDictMeter()
        # keep track of best val loss
        best_val_loss = np.inf
        for epoch in range(self.config["epochs"]):
            print(f"\nRunning Epoch {epoch + 1}/{self.config['epochs']}...")
            # training loop
            self.model.train()
            for batch in train_loader:
                # move batch to gpu
                batch = {k: v.to(self.config["device"]) for k, v in batch.items()}
                loss, predictions = self.train_step(batch, optimizer)
                train_metrics.update(predictions, batch["labels"])
                avg_loss_meter.add({"train_loss": loss})
                lr_scheduler.step()
                progress_bar.update(1)

            # validation loop
            self.model.eval()
            for batch in val_loader:
                batch = {k: v.to(self.config["device"]) for k, v in batch.items()}
                with torch.no_grad():
                    loss, predictions = self.val_step(batch)
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
                model_path = f"{self.save_model_dir}/{self.model_name}"
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                best_val_loss = mean_epoch_loss["val_loss"]
            if early_stopper.early_stop(mean_epoch_loss["val_loss"]):
                print("Stopping early")
                break

    def train_step(self, batch: dict, optimizer) -> Tuple[float, torch.Tensor]:
        """
        Performs a single training step with backpropagation
        :param optimizer: the optimizer to use
        :param batch: the batch to train on
        :return: the loss for the current batch and the predictions
        """
        # forward pass
        output = self.model(**batch)
        predictions = torch.argmax(output.logits, dim=-1)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), predictions

    def val_step(self, batch: dict) -> Tuple[float, torch.Tensor]:
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
        :return: a dataframe containing the predictions from the test data
        """
        progress_bar = tqdm(range(len(test_loader)))
        predictions = []
        metrics = TransformerMetricCollection(n_classes=self.config["num_labels"]).to(self.config["device"])
        self.model.eval()
        for batch in test_loader:
            batch = {k: v.to(self.config["device"]) for k, v in batch.items()}
            with torch.no_grad():
                _, preds = self.val_step(batch)
                metrics.update(preds, batch["labels"])
                for idx, pred in enumerate(preds.tolist()):
                    predictions.append({"y_hat_enc": pred, "y_enc": batch["labels"].flatten().tolist()[idx], })
            progress_bar.update(1)
        metrics = metrics.compute()
        print(metrics)
        wandb.log({"test": metrics})
        predictions = pd.DataFrame(predictions)
        return predictions
