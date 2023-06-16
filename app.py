import uvicorn
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.models.transformer import TransformerTandIClassifier

app = FastAPI()
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
    allow_headers=["*"], )


class Sample(BaseModel):
    text: str

@app.post("/predict")
def predict(samples: List[Sample]) -> str:
    model = TransformerTandIClassifier(model_name="distilbert-base-uncased", model_config={})
    inputs = [s.text for s in samples]
    encodings = model.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=512)
    predictions = model.predict(encodings)
    return predictions


if __name__ == "__main__":
    uvicorn.run(app)
