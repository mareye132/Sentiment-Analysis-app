import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset
# Make sure to replace 'your_dataset.csv' with your actual dataset file
# The dataset should have two columns: 'text' and 'label'
data = pd.read_csv('sentiment_data.csv')

# Check for missing values
data.dropna(subset=['text', 'label'], inplace=True)

# Features and labels
X = data['text']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer to disk
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")
# === Comment Categorization Training ===
cat_data = pd.read_csv('comment_categorization_dataset.csv')
cat_data.dropna(subset=['comment', 'category'], inplace=True)

X_cat = cat_data['comment']
y_cat = cat_data['category']

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
cat_vectorizer = TfidfVectorizer()
X_train_vec_cat = cat_vectorizer.fit_transform(X_train_cat)

from sklearn.naive_bayes import MultinomialNB
cat_model = MultinomialNB()
cat_model.fit(X_train_vec_cat, y_train_cat)

# Evaluate
X_test_vec_cat = cat_vectorizer.transform(X_test_cat)
y_pred_cat = cat_model.predict(X_test_vec_cat)
cat_accuracy = accuracy_score(y_test_cat, y_pred_cat)
print(f"Comment Categorization Model Accuracy: {cat_accuracy * 100:.2f}%")

# Save models
joblib.dump(cat_model, 'comment_category_model.pkl')
joblib.dump(cat_vectorizer, 'comment_category_vectorizer.pkl')

print("✅ Comment categorization model saved.")
