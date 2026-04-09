import streamlit as st
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Load model and tokenizer
model_clf = joblib.load('sentiment_model.joblib')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.squeeze() .cpu().numpy().reshape(1, -1)

st.title("Sentiment Classifier - BERT + Logistic Regression")
user_input = st.text_area("Enter a tweet or text:")

if st.button("Predict Sentiment"):
    embedding = get_bert_embedding(user_input)
    prediction = model_clf.predict(embedding)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    st.write(f"Sentiment: **{sentiment}**")
