import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Sentiment140 CSV (replace filename with your downloaded file path if different)
df = pd.read_csv('sentiment140.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Map sentiment 0=negative, 4=positive to 0/1 integer label
df['sentiment_label'] = df['target'].map({0: 0, 4: 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean tweets and remove stopwords
df['cleaned_text'] = df['text'].apply(clean_text)
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Save processed CSV for your pipeline
df[['cleaned_text', 'sentiment_label']].to_csv('cleaned_sentiment140.csv', index=False)

print("Sentiment140 preprocessing complete. Saved as cleaned_sentiment140.csv")
