import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

embeddings = np.load('bert_embeddings.npy', allow_pickle=True)
df = pd.read_csv("balanced_sentiment140.csv")
y = df['sentiment_label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'sentiment_model.joblib')
print("Model trained and saved as sentiment_model.joblib")
