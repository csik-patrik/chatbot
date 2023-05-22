
# Basic chatbot with python

## Description

This is a simple chatbot created using Python and TensorFlow that can understand and respond to user queries. The chatbot uses natural language processing techniques to identify the intent of the user's query and respond accordingly. The model is trained on a set of predefined intents and their corresponding patterns, which are stored in a JSON file.

## Requirements

- Python 3.x

- TensorFlow

- Keras

- NumPy

- NLTK

## Installation

- Clone the repository:

	- git clone https://github.com/csik-patrik/chatbot.git

- Navigate to the project directory:

- Install the required packages:

- Copy code

	- pip install -r "name of the package"

- Run the project:

	- python chatbot.py

## Usage

The chatbot prompts the user for input.

The user can input any query related to the predefined intents, such as greetings, farewells, or inquiries about products or services.

The chatbot will identify the intent of the query and respond accordingly.

## Training Data

The training data is a set of predefined intents and their corresponding patterns and responses, which are stored in a JSON file. The file includes various categories such as greetings, goodbye, weather, news, sports, music, and jokes. Each category has a unique tag associated with it, which is used by the chatbot to identify the user's query's intent.

For example, if the user inputs a query related to weather, the chatbot will use the 'weather' tag to identify the query's intent and respond accordingly. Each category includes a set of patterns that the chatbot uses to match the user's query. The patterns can be simple phrases like "hello" or complex sentences like "What's the weather like today?".

The chatbot's responses to the user's query are stored in the 'responses' field. The chatbot uses these responses to generate an appropriate answer based on the user's query. The responses can be simple one-liners like "Hello!" or more complex sentences providing more detailed information about the user's query.

This training data is used to train the chatbot model using natural language processing techniques. The model uses these patterns and responses to identify the user's intent and generate a response accordingly. The training data is an essential component of the chatbot as it enables the chatbot to understand and respond to the user's queries accurately.

The chatbot will identify the intent of the query and respond accordingly.

## Acknowledgements

The chatbot model is based on the tutorial: 

	https://www.youtube.com/watch?v=1lwddP0KUEg

The data for training the model is taken from the same tutorial.