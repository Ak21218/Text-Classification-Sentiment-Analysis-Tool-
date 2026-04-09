import pandas as pd
from transformers import pipeline

# Load cleaned tweet text
df = pd.read_csv('cleaned_tweets.csv')

# Use Hugging Face's sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Get labels from the model
results = sentiment_analyzer(df['cleaned_text'].tolist(), truncation=True)

df['sentiment_label'] = [res['label'] for res in results]
df['sentiment_score'] = [res['score'] for res in results]

df.to_csv("pseudo_labeled_tweets.csv", index=False)
print("Pseudo-labeling complete. Labels saved to pseudo_labeled_tweets.csv")
