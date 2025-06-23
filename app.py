# app.py
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("🧠 Sentiment Analysis Web App")

# User input
user_input = st.text_area("Enter a sentence:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Show result
        sentiment = "😊 Positive" if prediction == 1 else "😠 Negative"
        st.subheader(f"Sentiment: {sentiment}")
