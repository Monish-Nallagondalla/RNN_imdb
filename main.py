# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    # Split the text into lowercase words
    words = text.lower().split()
    # Map words to their respective indices, using 2 (unknown token) for unknown words
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Pad the sequence to match the input length for the model
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit App for Sentiment Analysis
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    # Preprocess the input
    preprocessed_input = preprocess_text(user_input)

    # Make the prediction
    prediction = model.predict(preprocessed_input)
    score = prediction[0][0]  # Extract the prediction score
    sentiment = 'Positive' if score > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
    
    # Debugging: Display preprocessed input (optional)
    st.write(f'Preprocessed Input: {preprocessed_input}')
else:
    st.write('Please enter a movie review.')


#