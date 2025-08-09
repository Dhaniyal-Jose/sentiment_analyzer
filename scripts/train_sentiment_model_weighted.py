import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.utils import class_weight

# Load the dataset
df = pd.read_csv('C:/Users/dhani/Downloads/sentiment_analyzer/data/sentiment_analysis.csv')

# Preprocess the data
sentences = df['content'].values
labels = df['sentiment'].values

# Encode the labels
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(integer_labels)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(integer_labels), y=integer_labels)
class_weights_dict = dict(enumerate(class_weights))

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, categorical_labels, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(categorical_labels.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with class weights
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)

# Save the model
model.save('C:/Users/dhani/Downloads/sentiment_analyzer/models/sentiment_model_weighted.h5')

import pickle

# Save the tokenizer
with open('C:/Users/dhani/Downloads/sentiment_analyzer/models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the label encoder
with open('C:/Users/dhani/Downloads/sentiment_analyzer/models/label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model training complete and saved to C:/Users/dhani/Downloads/sentiment_analyzer/models/sentiment_model_weighted.h5")
print("Tokenizer and label encoder saved.")
