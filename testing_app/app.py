import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax


MODEL_DIR = '/checkpoint/2024-07-30 22:10:25-distilbert-base-uncased/seed1'
TOKENIZER_DIR = '/checkpoint/2024-07-30 22:10:25-distilbert-base-uncased'


def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    model.eval()  
    return model, tokenizer


def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = softmax(outputs.logits, dim=1)
    return probabilities


model, tokenizer = load_model()


st.title("TandI-Classifier")
user_input = st.text_input("Enter text for classification:")
if st.button("Submit"):
    if user_input:
        probabilities = predict(user_input, model, tokenizer)
        st.write("Classification probabilities:")
        for idx, prob in enumerate(probabilities.squeeze(0)):
            st.write(f"Class {idx}: {prob.item():.6f}")
    else:
        st.error("Please enter some text to classify.")
