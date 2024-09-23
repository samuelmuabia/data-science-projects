import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, Flatten, Dense

# Set up paths
relativePath = os.getcwd()
dataPath = relativePath + "/Text Emotion Classification/datasets/train.txt"

# Load dataset
data = pd.read_csv(dataPath, sep=';')
data.columns = ["Text", "Emotions"]

# Extract texts and labels
texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
one_hot_labels = keras.utils.to_categorical(labels)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, one_hot_labels, test_size=0.2)

# Check if the model already exists to avoid retraining
if os.path.exists('emotion_model.h5'):
    # Load the saved model
    print("Loading existing model...")
    model = load_model('emotion_model.h5')

    # Load tokenizer and label encoder
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
else:
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
                        output_dim=128, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

    # Save the trained model
    model.save('emotion_model.h5')

    # Save the tokenizer and label encoder
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

# Make a prediction on new text
input_text = "i am ever feeling nostalgic about the fireplace i will know that it is still on the property"
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)

# Predict the emotion
prediction = model.predict(padded_input_sequence)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
print(f"Predicted Emotion: {predicted_label[0]}")
