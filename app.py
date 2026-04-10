import streamlit as st
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os

# Set HuggingFace cache directory
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/hub')

# Load model and tokenizer
model_clf = joblib.load('sentiment_model.joblib')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768).reshape(1, -1)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.pooler_output.detach().cpu().numpy()
    return embedding.reshape(1, -1) if embedding.ndim == 1 else embedding

st.title("Sentiment Analyzer")
user_input = st.text_area("Enter a tweet or text:")

if st.button("Analyze Sentiment"):
    embedding = get_bert_embedding(user_input)
    proba = model_clf.predict_proba(embedding)[0]
    negative_score = float(proba[0])
    positive_score = float(proba[1])

    if positive_score >= 0.70:
        sentiment = "Positive"
    elif positive_score <= 0.30:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    st.write(f"Sentiment: **{sentiment}**")
    st.write(f"Confidence: Negative={negative_score:.2f}, Positive={positive_score:.2f}")
