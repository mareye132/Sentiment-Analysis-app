# train_model.py
# importing neccessary liberaries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample training data
texts = [
    "I love this product",
    "This is amazing",
    "I hate this",
    "Worst experience ever",
    "Excellent service",
    "Very bad"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Convert text to numeric features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train classifier
model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved.")
