import torch
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from models.transformer import TransformerTandIClassifier

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class Sample(BaseModel):
    text: str


@app.post("/predict")
def predict(samples: List[Sample]) -> List[str]:
    """
    entrypoint for predicting the labels of a list of samples
    :param samples: the samples to predict
    :return: a list with a label for each sample
    """
    if os.path.exists(f"app/checkpoints/distilbert-base-uncased"):
        model = TransformerTandIClassifier(
            model_name="distilbert-base-uncased",
            # TODO: create a default config somewhere else instead of hardcoding it here
            model_config={
                "device": 'cuda' if torch.cuda.is_available() else 'cpu',
                "num_labels": 17
            }
        )
        samples = [s.text for s in samples]
        predictions = model.predict(samples)
        return predictions
    else:
        raise Exception("No model checkpoint found. Please train a model first.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
