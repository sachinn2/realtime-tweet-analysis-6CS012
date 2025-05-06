import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load models
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model_paths = {
    "RNN Model 1": "sachin_simple_rnn_model.h5",
    "RNN Model 2": "sachin_lstm_1_model.h5",
    "RNN Model 3": "sachin_lstm_2_model.h5"
}

# UI
st.title("Tweet Racism/Sexism Detector")
st.write("Enter a tweet and select a model to predict if it's racist/sexist.")

# Input field
user_input = st.text_area("Tweet Text", placeholder="Type your tweet here...")

# Model selector
selected_model_name = st.selectbox("Choose a Model", list(model_paths.keys()))

# Button
if st.button("Check Tweet"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        # Tokenize and pad
        sequences = tokenizer.texts_to_sequences([user_input])
        padded_input = pad_sequences(sequences, maxlen=100)  # Change if your model uses different maxlen

        # Load selected model
        model = load_model(model_paths[selected_model_name])
        prediction = model.predict(padded_input)[0][0]

        label = "🚫 Racist/Sexist" if prediction >= 0.5 else "✅ Not Racist/Sexist"
        st.subheader("Prediction:")
        st.success(label)
