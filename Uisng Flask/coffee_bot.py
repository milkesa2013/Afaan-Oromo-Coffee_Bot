# for tokenizing text
from tensorflow.keras.preprocessing.text import Tokenizer
# for lemmatization
from nltk.stem import WordNetLemmatizer
# for handling word2vec models
from gensim.models import KeyedVectors
# for loading saved data structures 
import pickle
# for handling numerical operations
import numpy as np
# for handling data in a DataFrame
import pandas as pd
# importing load_model to load the pre-trained Keras model
from keras.models import load_model
# for hanling dataset
import json
# to help random response
import random
# Import regular expressions for text preprocessing
import re
# importing word_tokenize to tokenize sentences into words
from nltk.tokenize import word_tokenize
# for encoding target labels as integers
from sklearn.preprocessing import LabelEncoder
# for stop word removal
from nltk.corpus import stopwords
# importing Flask modules to build a web application
from flask import Flask, render_template, request
# importing pad_sequences for input lenght standardization
from tensorflow.keras.preprocessing.sequence import pad_sequences

# creating a Flask application instance
app = Flask(__name__)
# Setting the static folder for serving static files
app.static_folder = 'static'
# loading pre-trained word vectors
word_vectors = KeyedVectors.load("word vectors path/word2vec_model.bin")
# loading trained model
model = load_model('model path/protbot.h5')
# loading stop words for Afaan Oromoo language
Ao_stopwords = stopwords.words('afaan_oromo')
# defining ignored words
ignore_words = ['?',']','[','#','@','!','$','^','&','*','.','>','<','}','{','|','/','~','+','-','=',':',';',',']
# loading the dataset
intents = json.loads(open('dataset path/coffeedata.json', "r", encoding='utf-8').read())
# loading preprocessed words and classes from saved files
words = pickle.load(open("path for preprocessed words/filtered_text.pkl", "rb"))
classes = pickle.load(open("path for classes saved/classess.pkl", "rb"))

# Extract intents and responses from the loaded JSON file
tags = []  # List to store all the tags (intent categories)
inputs = []  # List to store all the input patterns
responses = {}  # Dictionary to map each tag to its corresponding responses
for intent in intents['intents']:
    # Store responses for each tag
    responses[intent['tag']] = intent["responses"]
    for lines in intent['patterns']:
        # Append input patterns (user queries)
        inputs.append(lines)
        # Append corresponding tag for each pattern
        tags.append(intent['tag'])

# Creating a pandas DataFrame to store inputs and their corresponding tags
data = pd.DataFrame({"inputs": inputs, "tags": tags})
# tokenizing the sentences 
sentences = [word_tokenize(text) for text in data["inputs"].tolist()]

# Preprocess tokenized sentences by removing special characters, stopwords, and lowercasing
filtered_sentences = []  # List to store filtered sentences
for sentence in sentences:
    filtered_sentence = []  # Temporary list to store words in each sentence
    for word in sentence:
        # Convert word to lowercase
        word = word.lower()
        # Remove non-alphanumeric characters
        word = re.sub(r'[^a-zA-Z0-9]', '', word)
        # Remove stopwords
        if word not in Ao_stopwords:
            # Add the cleaned word to the filtered sentence
            filtered_sentence.append(word)
    # Append filtered sentence to the list
    filtered_sentences.append(filtered_sentence)

# Fit the tokenizer on the filtered sentences and convert words to their index values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_sentences)
# Convert sentences to sequences of word indices
train = tokenizer.texts_to_sequences(filtered_sentences)
# Pad the sequences to ensure consistent input shape
x_train = pad_sequences(train)
# Label encoding the tags (intent classes) into integers
le = LabelEncoder()
# Transform string tags to integers
y_train = le.fit_transform(data["tags"])
# Determine input shape based on padded sequences
input_shape = x_train.shape[1]

# Initialize WordNet lemmatizer for reducing words to their base form
lemmatizer = WordNetLemmatizer()
# Function to preprocess user input by cleaning it (removing stopwords, special characters, etc.)
def clean_up_sentence(sentence):
    # Assign user input to a variable
    user_input = sentence
    # List to store processed input
    prediction_input = []
    for letter in user_input:
        # Ignore certain characters and stopwords
        if letter not in ignore_words and letter not in Ao_stopwords:
            # Lowercase the letter
            letter = letter.lower()
            # Append cleaned letter to the list
            prediction_input.append(letter)
    return prediction_input

# Function to generate word embedding for a sentence
def word_emb(sentence, words, show_details=True):
    # List to store processed sentence
    textList = []
    # Clean the sentence
    prediction_input = clean_up_sentence(sentence)
    # Join cleaned characters to form a string
    prediction_input = ''.join(prediction_input)
    # Add the cleaned text to the list
    textList.append(prediction_input)
    return textList

# Function to predict the intent class based on user input
def predict_class(patterns, model, threshold=0.75):
    # Process the input sentence
    p = word_emb(patterns, words, show_details=False)
    # Convert sentence to sequence
    predic = tokenizer.texts_to_sequences(p)
    # Reshape input for prediction
    predic = np.array(predic).reshape(-1)
    # Pad sequence to match input shape
    predic = pad_sequences([predic], input_shape)
    # Predict using the trained model
    output = model.predict(predic)
    # Set prediction threshold
    threshold = 0.75
    # Check if the prediction confidence exceeds threshold
    if np.max(output) >= threshold:
        # Get the predicted class
        output = output.argmax()
        # Convert predicted class index to label
        response_tag = le.inverse_transform([output])[0]
        # Choose a random response for the predicted class
        res = random.choice(responses[response_tag])
    else:
        # Default response for low confidence
        res = "Dhiifama waan isin jettan naaf hin galle; yaada keessan ifa naa godhaa"
    return res

# Function to get the chatbot response 
def chatbot_response(msg):
    # Get predicted class and response
    ints = predict_class(msg, model)
    return ints

# Define the homepage route
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for getting the bot's response
@app.route("/get")
def get_bot_response():
    # Get user message from request
    userText = request.args.get('msg')
    # Get and return chatbot response
    return chatbot_response(userText)

# Run the Flask application 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
