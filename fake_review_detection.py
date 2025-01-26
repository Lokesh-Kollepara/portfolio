# fake_review_detection.py
import streamlit as st
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# File Configuration
MODEL_KEYS = {
    'svm': 'svm_model.pkl',
    'voting': 'voting_model.pkl'
}
VECTORIZER_PATH = 'vectorizer.pkl'

# Check for model and vectorizer files
for model_path in MODEL_KEYS.values():
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found in the current directory.")
        st.stop()

if not os.path.exists(VECTORIZER_PATH):
    st.error(f"Vectorizer file {VECTORIZER_PATH} not found in the current directory.")
    st.stop()

# Load the models and vectorizer
models = {name: joblib.load(path) for name, path in MODEL_KEYS.items()}
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

# Define the preprocessing function
def text_preprocessing(text):
    stop_words = set(stopwords.words('english'))
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(text))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    cleaned = [token for token in tokens if token not in stop_words]
    stemmed = [stemmer.stem(token) for token in cleaned]
    return " ".join(stemmed)

def run_fake_review_detection():
    """Streamlit app for Fake Review Detection."""
    st.header("Fake Review Detection")

    # Input fields
    review_text = st.text_area("Enter the Review Text")
    model_choice = st.selectbox("Select a Model", options=list(models.keys()), index=0)

    if st.button("Detect Review Type"):
        if review_text:
            try:
                # Preprocess the review text
                processed_review = text_preprocessing(review_text)

                # Transform the text using the vectorizer
                transformed_text = tfidf_vectorizer.transform([processed_review])

                # Predict the label using the selected model
                prediction = models[model_choice].predict(transformed_text)
                label = "CG" if prediction[0] == 0 else "OR"

                # Display the result
                st.success(f"The review is classified as: {label}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a review text.")
