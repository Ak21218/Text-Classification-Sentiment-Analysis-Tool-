
import pandas as pd

df = pd.read_csv("cleaned_sentiment140.csv")
neg = df[df['sentiment_label'] == 0].sample(500, random_state=42)
pos = df[df['sentiment_label'] == 1].sample(500, random_state=42)
df_balanced = pd.concat([neg, pos]).sample(frac=1, random_state=42)  # Shuffle
df_balanced.to_csv("balanced_sentiment140.csv", index=False)
print("Saved balanced sample to balanced_sentiment140.csv")
