import nltk
import torch
import logging
from typing import List
from pickle import load
from collections import Counter
from transformers import AutoTokenizer
from transformers import DistilBertForSequenceClassification
from ..inference.models import ArticleChunk, ChunkResult, ArticleResult, Prediction

MIN_CHUNK_CONFIDENCE = 0.5

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
        self.model.eval()
        self.le = load(open(f'checkpoints/{self.model_name}/label_encoder.pkl', 'rb'))
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

    def predict(self, article_chunks: List[ArticleChunk]):
        """
        computes the class probabilities for each article chunk
        :param article_chunks: a list of article chunks
        :return: the article chunks with the class probabilities
        """
        chunk_results = []
        encoded_chunks = self.tokenizer(
            [chunk.text for chunk in article_chunks],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        encoded_chunks = encoded_chunks.to(self.config["device"])
        with torch.no_grad():
            output = self.model(**encoded_chunks, output_attentions=False, output_hidden_states=False)
            class_probs_per_chunk = torch.nn.Softmax(dim=1)(output.logits)
        for chunk_idx, article_chunk in enumerate(article_chunks):
            chunk_results.append(ChunkResult(
                chunk=article_chunk,
                class_probabilities=[Prediction(
                    class_label=self.le.inverse_transform([class_idx]).item(),
                    probability=prob
                ) for class_idx, prob in enumerate(class_probs_per_chunk[chunk_idx])]
            ))
        most_probable_class_per_chunk = torch.argmax(class_probs_per_chunk, dim=1).tolist()
        most_probable_class_per_chunk = [c if class_probs_per_chunk[i][c] > MIN_CHUNK_CONFIDENCE else -1 for i, c in enumerate(most_probable_class_per_chunk)]
        counter = Counter(most_probable_class_per_chunk)
        most_occurring_class, count = counter.most_common(1)[0]
        return ArticleResult(
            article_id=article_chunks[0].article_id,
            chunk_predictions=chunk_results,
            class_label=self.le.inverse_transform([most_occurring_class]).item() if most_occurring_class != -1 else "unknown",
            probability=count/len(article_chunks) if most_occurring_class != -1 else 0
        )
