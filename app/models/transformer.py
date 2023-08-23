from transformers import AutoTokenizer
import torch
import logging
from typing import List
from pickle import load
from transformers import DistilBertForSequenceClassification


class TransformerTandIClassifier():
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
        logging.info("Loading preexisting model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            f"checkpoints/{self.model_name}",
            local_files_only=True
        ).to(self.config["device"])
        tokenizer = AutoTokenizer.from_pretrained(f"checkpoints/{self.model_name}", local_files_only=True)
        logging.info("Model loaded.")
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

        le = load(open(f'checkpoints/{self.model_name}/label_encoder.pkl', 'rb'))
        return list(le.inverse_transform(predictions.tolist())), probs
