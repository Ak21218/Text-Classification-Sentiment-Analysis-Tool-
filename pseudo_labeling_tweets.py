import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load embeddings and labels
embeddings = np.load('bert_embeddings.npy')
labels = pd.read_csv('pseudo_labeled_tweets.csv')['sentiment_label']

# Convert labels to numeric
label_map = {'NEGATIVE': 0, 'POSITIVE': 1}
y = labels.map(label_map).values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(clf, 'sentiment_bert_classifier.joblib')
print("Model saved as sentiment_bert_classifier.joblib.")




#Loads embeddings and labels,
#Splits dataset into train/test,
#Trains logistic regression,
#Prints classification report,
#Saves the model for future prediction or deployment.