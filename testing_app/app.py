import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax
import os
import joblib

print(os.getcwd())


# TODO: ADJUST CHECKPOINT and SEED
CHECKPOINT = '2024-07-30 22:10:25-distilbert-base-uncased'
SEED = 'seed_1'

MODEL_DIR = f'./checkpoint/{CHECKPOINT}/{SEED}/'
TOKENIZER_DIR = f'./checkpoint/{CHECKPOINT}'

def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    label_encoder = joblib.load(f"{TOKENIZER_DIR}/label_encoder.pkl") 
    model.eval()
    return model, tokenizer, label_encoder

def predict(text, model, tokenizer, label_encoder):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = softmax(outputs.logits, dim=1).squeeze(0)
    # Convert logits to class names with probabilities
    results = [(label_encoder.inverse_transform([i])[0], prob.item()) for i, prob in enumerate(probabilities)]
    # Sort results by probability in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results


model, tokenizer, label_encoder = load_model()


st.title("TandI-Classifier")
user_input = st.text_input("Enter text for classification:")
if st.button("Submit"):
    if user_input:
        sorted_results = predict(user_input, model, tokenizer, label_encoder)
        st.write("Classification probabilities:")
        for name, prob in sorted_results:
            if prob == sorted_results[0][1]: 
                st.markdown(f"**{name}: {prob:.6f}**", unsafe_allow_html=True)  
            else:
                st.write(f"{name}: {prob:.6f}")
    else:
        st.error("Please enter some text to classify.")
