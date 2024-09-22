import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# modelPath = os.path.join(relativePath, 'path/to/emotion_model.h5')
# model = load_model(modelPath)

# Load the saved model, tokenizer, and label encoder
@st.cache_resource
def load_model_resources():
    model = load_model('emotion_model.h5')
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

# Preprocess the input text and predict emotion
def predict_emotion(input_text, model, tokenizer, label_encoder):
    # Convert input text to sequence
    input_sequence = tokenizer.texts_to_sequences([input_text])
    max_length = model.input_shape[1]  # Get the max length from the model input shape
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    
    # Predict the emotion
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    
    return predicted_label[0]

# Streamlit UI
st.title("Text Emotion Classifier")

st.write("Enter a text to find out its emotion!")

# Get user input
input_text = st.text_input("Enter your text:")

# Load model resources
model, tokenizer, label_encoder = load_model_resources()

# Display prediction
if st.button("Predict Emotion"):
    if input_text.strip() != "":
        predicted_emotion = predict_emotion(input_text, model, tokenizer, label_encoder)
        st.write(f"The predicted emotion is: **{predicted_emotion}**")
    else:
        st.write("Please enter some text.")
