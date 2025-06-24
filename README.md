# Sentiment-Analysis-app
## 🔍 Sentiment Analysis Web App

This is a Streamlit-based web app that uses a Logistic Regression model to classify text sentiment as **Positive** or **Negative**.

### 🧠 Features:
- Trained on 1000 synthetic samples (500 positive, 500 negative)
- TF-IDF vectorization of input text
- 100% accuracy on a 300-sample test set
- Simple and clean UI for real-time sentiment prediction

### 🚀 Tech Stack:
- Python, Scikit-learn, Streamlit
- Model serialized using Joblib
Project Structure
sentiment_app/
├── app.py               # Streamlit app interface
├── train_model.py       # ML training script
├── sentiment_model.pkl  # Trained sentiment model
├── vectorizer.pkl       # TF-IDF vectorizer
└── requirements.txt     # Dependencies
