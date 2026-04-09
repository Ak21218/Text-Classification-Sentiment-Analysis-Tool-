import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load cleaned tweets
# Sample 1000 rows for fast testing
df = pd.read_csv("balanced_sentiment140.csv")



# Fill missing cleaned_text with empty string
df['cleaned_text'] = df['cleaned_text'].fillna("")



# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768)  # Return zero vector for invalid or empty strings
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze().numpy()

# Generate embeddings for each tweet
embeddings = df['cleaned_text'].apply(get_bert_embedding)

# Convert embeddings list to 2D array and save
embeddings_matrix = np.stack(embeddings.values)

np.save('bert_embeddings.npy', embeddings_matrix)

print("BERT embeddings generated and saved to bert_embeddings.npy")
