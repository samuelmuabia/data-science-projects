import keras
import tensorflow
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Load the model
model = load_model('emotion_model.h5')

# Now you can use the model for prediction
input_text = "i am ever feeling nostalgic about the fireplace i will know that it is still on the property"
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
prediction = model.predict(padded_input_sequence)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
print(predicted_label[0])
