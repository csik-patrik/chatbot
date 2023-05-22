# Import necessary libraries
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

# Create a WordNetLemmatizer object
# Learn more about the nltk.stem.wordnet module:
# https://www.nltk.org/api/nltk.stem.wordnet.html
lemmatizer = WordNetLemmatizer()

# Load intents JSON data into 'intents' dictionary
intents = json.loads(open('../data/intents.json').read())

# Initialize lists and variables
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Iterate over the 'patterns' in each 'intent' to extract 'words' and 'classes'
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and filter the words to remove punctuation and standardize the format
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words))

classes = sorted(set(classes))

# Save the words and classes to disk for future use
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))

# Initialize training data lists and output empty list
training = []
output_empty = [0] * len(classes)

# Iterate over the 'documents' to create training data
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1 
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Convert the training data to numpy arrays
training = np.array(training)
train_x = np.array([row[0] for row in training], dtype=np.float32)
train_y = np.array([row[1] for row in training], dtype=np.float32)

# Build the neural network model

# The network architecture consists of an input layer, two hidden layers, and an output layer.
# The input layer has a number of neurons equal to the number of features in the training data (len(train_x[0])).
# The two hidden layers are densely connected layers with 128 and 64 units, respectively, and ReLU activation function.
# Dropout regularization with a rate of 0.5 is applied to the hidden layers to prevent overfitting.
# The output layer has units equal to the number of classes, and a softmax activation function is applied to produce probabilities for each class.

# Create a sequential model using Keras
model = Sequential()

# Add a densely connected layer with 128 units, ReLU activation, and input shape of (len(train_x[0]),)
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Add dropout regularization with a rate of 0.5
model.add(Dropout(0.5))
# Add another densely connected layer with 64 units and ReLU activation
model.add(Dense(64, activation='relu'))
# Add dropout regularization with a rate of 0.5
model.add(Dropout(0.5))
# Add the output layer with units equal to the number of classes and softmax activation
model.add(Dense(len(classes), activation='softmax'))

# Set optimizer and compile the model
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model and save it to disk
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('The training is done!')
