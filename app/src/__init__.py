import os
import torch
from .models.transformer import (TransformerTandIClassifier)

if os.path.exists(f"checkpoints/distilbert-base-uncased"):
    MODEL = TransformerTandIClassifier(
        model_name="distilbert-base-uncased",
        model_config={
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "num_labels": 17
        })
else:
    raise Exception("No model checkpoint found. Please train a model first.")