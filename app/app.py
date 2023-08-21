import torch
import os
import uvicorn
import logging
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from models.transformer import TransformerTandIClassifier
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
logging.basicConfig(level=logging.INFO)
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

if os.path.exists(f"checkpoints/distilbert-base-uncased"):
    MODEL = TransformerTandIClassifier(
        model_name="distilbert-base-uncased",
        model_config={
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "num_labels": 17
        })
else:
    raise Exception("No model checkpoint found. Please train a model first.")


class Sample(BaseModel):
    text: str


@app.post("/predict")
def predict(samples: List[Sample]) -> List[str]:
    """
    entrypoint for predicting the labels of a list of samples
    :param samples: the samples to predict
    :return: a list with a label for each sample
    """
    samples = [s.text for s in samples]
    predictions, probs = MODEL.predict(samples)
    return predictions


@app.get("/")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('app.html', context={'request': request, 'result': result})


@app.post("/")
def form_post(request: Request, text: str = Form(...)):
    prediction, probs = MODEL.predict([text])
    result = f"category: {prediction[0]} ({round(probs[0].item(), 2)})"
    return templates.TemplateResponse('app.html', context={'request': request, 'result': result})


#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
