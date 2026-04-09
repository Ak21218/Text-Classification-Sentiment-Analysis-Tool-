import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load model and test data
model = joblib.load('sentiment_model.joblib')
embeddings = np.load('bert_embeddings.npy', allow_pickle=True)
df = pd.read_csv('balanced_sentiment140.csv')
y = df['sentiment_label'].values

# Split same way as during training
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42
)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
