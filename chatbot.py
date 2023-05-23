import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Initialize the WordNet lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Load the intents data from the JSON file
intents = json.loads(open('../data/intents.json').read())

# Load the preprocessed words and classes using pickle
words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))

# Load the trained chatbot model
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    # Tokenize the sentence into individual words
    sentence_words = nltk.word_tokenize(sentence)

    # Lemmatize each word to its base form
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # Clean up the sentence by lemmatizing and tokenizing the words
    sentence_words = clean_up_sentence(sentence)

    # Create a bag-of-words representation of the sentence
    bag = [0] * len(words)
    for w in sentence_words:
        # Set the index of the word in the bag to 1 if it is present in the sentence
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Convert the sentence into a bag-of-words representation
    bow = bag_of_words(sentence)

    # Make predictions using the loaded model
    res = model.predict(np.array([bow]))[0]

    # Set a threshold to filter out low-probability intents
    ERROR_THRESHOLD = 0.25

    # Get the intents and probabilities above the threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort the results by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Create a list of intents and their probabilities
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    # Get the predicted intent from the list
    tag = intents_list[0]['intent']

    # Retrieve the list of intents from the intents JSON data
    list_of_intents = intents_json['intents']

    # Find the matching intent and select a random response
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Hi! How can I help you?')

while True:
    # Get user input
    message = input("")

    # Predict the intent based on the input message
    ints = predict_class(message)

    # Get a response based on the predicted intent
    res = get_response(ints, intents)

    # Print the response
    print(res)
