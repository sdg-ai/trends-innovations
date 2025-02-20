import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax
import os
import joblib

class TIClassifier:
    def __init__(self, checkpoint_dir, seed):
        self.model_dir = os.path.join(checkpoint_dir, seed)
        self.tokenizer_dir = checkpoint_dir
        self.model = None
        self.tokenizer = None
        self.label_encoder = None

    def load_model(self):
        """Load the model, tokenizer and label encoder"""
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir, local_files_only=True)
        self.label_encoder = joblib.load(os.path.join(self.tokenizer_dir, "label_encoder.pkl"))
        self.model.eval()
        return self

    def predict(self, text):
        """Predict the class for the input text"""
        if not self.model or not self.tokenizer or not self.label_encoder:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = softmax(outputs.logits, dim=1).squeeze(0)
        results = [(self.label_encoder.inverse_transform([i])[0], prob.item()) 
                  for i, prob in enumerate(probabilities)]
        return sorted(results, key=lambda x: x[1], reverse=True)