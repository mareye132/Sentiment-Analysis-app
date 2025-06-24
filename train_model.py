# train_model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# 📂 Load dataset from CSV
data = pd.read_csv("sentiment_data.csv")  # Make sure the file is in the same directory

texts = data["text"].tolist()
labels = data["label"].tolist()

# 🔀 Split into training and testing sets (stratified to balance classes)
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# ✨ Feature extraction
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# 🤖 Train classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# 🧪 Evaluate performance
y_pred = model.predict(X_test)
print("\n📊 Model Performance on Test Set:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0))

# 💾 Save the trained model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model trained and saved.")
