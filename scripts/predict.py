import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from utils.preprocess import clean_text

# Load the trained model
model = load_model('C:/Users/dhani/Downloads/sentiment_analyzer/models/sentiment_model_weighted.h5')

# Load the tokenizer
with open('C:/Users/dhani/Downloads/sentiment_analyzer/models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('C:/Users/dhani/Downloads/sentiment_analyzer/models/label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Prepare a sample sentence
sample_sentence = "I love this product! It's amazing."

# Preprocess the sentence
cleaned_sentence = clean_text(sample_sentence)
sequence = tokenizer.texts_to_sequences([cleaned_sentence])
padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

# Predict the sentiment
prediction = model.predict(padded_sequence)
predicted_label_index = np.argmax(prediction)
predicted_label = label_encoder.inverse_transform([predicted_label_index])

print(f"Sentence: {sample_sentence}")
print(f"Predicted sentiment: {predicted_label[0]}")