import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reversed_word_index = {value: key for key,value in word_index.items()}

# from keras.saving.legacy_h5_format import load_model as legacy_load_model

model = load_model('simple_rnn_imdb.h5')

## 2. helper function
## function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3,'?') for i in encoded_review])

## function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## Prediction Function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]


import streamlit as st

st.title("IMDB Moview Review Sentiment Analysis")
st.write("Enter a Movie Review to classify it as positibe or negative")

## user input

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
    
    ## make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    st.write(f"Sentiment : {sentiment}")
    st.write(f"Prediction Score : {prediction[0][0]}")
else:
    st.write("Please Enter a moview review")    




