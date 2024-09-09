import logging 
import json  
import random  
import re  
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
import pickle  
import numpy as np 
import pandas as pd  
from tensorflow.keras.preprocessing.text import Tokenizer  
from gensim.models import KeyedVectors  
from sklearn.preprocessing import LabelEncoder  
from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from telegram import Update  
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters  

# Setting up logging for the application
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)  # Create a logger to log info and debug messages

# Loading pre-trained Word2Vec model 
word_vectors = KeyedVectors.load("word2vec modelpath/word2vec_model.bin")

# Loading the trained chatbot model 
model = load_model('saved_model path/protbot.h5')

# Loading Afaan Oromoo stopwords 
Ao_stopwords = stopwords.words('afaan_oromo')

# List of symbols/characters to be ignored during text preprocessing
ignore_words = ['?',']','[','#','@','!','$','^','&','*','.','>','<','}','{','|','/','~','+','-','=',':',';',',']

#loading the dataset from the computer
intents = json.loads(open('dataset_path/coffeedata.json',"r",encoding='utf-8').read())

# Loading preprocessed words and classes and words
words = pickle.load(open("saved_preprocessed_datapath/filtered_text.pkl", "rb"))
classes = pickle.load(open("saved classes path/classess.pkl", "rb"))

# Initialize lists to store input patterns and their corresponding tags 
tags = []  # Stores the different intent tags
inputs = []  # Stores the user input patterns
responses = {}  # Dictionary to map each tag to its list of responses

# Iterate through each intent in the intents JSON data
for intent in intents['intents']:
    responses[intent['tag']] = intent["responses"]  # Map the tag to the corresponding responses
    for lines in intent['patterns']:  # Loop through each pattern (user query)
        inputs.append(lines)  # Append the pattern to inputs list
        tags.append(intent['tag'])  # Append the corresponding tag

# Create a Pandas DataFrame to organize inputs and tags
data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Tokenize each input sentence 
sentences = [word_tokenize(text) for text in data["inputs"].tolist()]

# Initialize a list to store the cleaned and filtered sentences
filtered_sentences = []
for sentence in sentences:
    filtered_sentence = [] 
    for word in sentence:
        word = word.lower()  # normalizing by  lowercasing
        word = re.sub(r'[^a-zA-Z0-9]', '', word)  # Remove non-alphanumeric characters
        if word not in Ao_stopwords:  # Remove stopwords
            filtered_sentence.append(word)  # Append cleaned word to the list
    filtered_sentences.append(filtered_sentence)  # Add filtered sentence to the main list

# Initialize the Tokenizer and fit it on the filtered sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_sentences)  # Fit tokenizer on the word list
train = tokenizer.texts_to_sequences(filtered_sentences)  # Convert text to word index sequences

# Pad sequences to ensure they all have the same length
x_train = pad_sequences(train)

# Encode the target labels 
le = LabelEncoder()
y_train = le.fit_transform(data["tags"])

# Determine the input shape for the model based on the padded sequences
input_shape = x_train.shape[1]

# Function to clean and preprocess user input
def clean_up_sentence(sentence):
    user_input = sentence  # Get user input
    prediction_input = []  # List to store cleaned input
    for letter in user_input:
        if letter not in ignore_words and letter not in Ao_stopwords:  # Filter out stopwords and special characters
            letter = letter.lower()  # Convert to lowercase
            prediction_input.append(letter)  # Add cleaned letter to the list
    return prediction_input  # Return the cleaned sentence

# Define maximum sequence length for padding
max_sequence_length = 16

# Function to embed words and return a cleaned version of the sentence
def word_emb(sentence, words, show_details=True):
    textList = []  # Initialize a list to store processed sentences
    prediction_input = clean_up_sentence(sentence)  # Clean the input sentence
    prediction_input = ''.join(prediction_input)  # Join the cleaned words
    textList.append(prediction_input)  # Append the cleaned sentence
    return textList  # Return the processed sentence

# Function to predict the class  user input
def predict_class(patterns, model, threshold=0.5):
    p = word_emb(patterns, words, show_details=False)  # Embed the input sentence
    predic = tokenizer.texts_to_sequences(p)  # Convert the sentence into a sequence of word indices
    predic = np.array(predic).reshape(-1)  # Reshape the sequence for model input
    predic = pad_sequences([predic], input_shape)  # Pad the sequence to match input shape
    output = model.predict(predic)  # Get the prediction from the model
    threshold = 0.75  # Set a threshold to filter weak predictions
    if np.max(output) >= threshold:  # If the prediction exceeds the threshold
        output = output.argmax()  # Get the index of the predicted class
        response_tag = le.inverse_transform([output])[0]  # Convert the index back to the tag
        res = random.choice(responses[response_tag])  # Choose a random response from the predicted class
    else:
        res = "Dhiifama waan isin jettan naaf hin galle; yaada keessan ifa naa godhaa"  # Default response for unclear input
    return res  # Return the response

# Function to get chatbot's response for a given message
def chatbot_response(msg):
    ints = predict_class(msg, model)  # Predict the class for the input message
    return ints  # Return the response

# Function to start the bot and send a greeting message when the user initiates a conversation
def start(update: Update, context: CallbackContext):
    user = update.effective_user  # Get the user's information
    update.message.reply_text(f"Harka fuune {user.first_name}! Ani Bot oomisha bunaa irratti gargaarsa yaadaa kennuuf qopha'eedha.Oomisha bunaan walqabatee gaaffii yoo qabaattan na gaafadhaa!")  # Reply with a greeting message

# Function to handle messages from the user and respond using the chatbot
def echo(update: Update, context: CallbackContext):
    message = update.message.text  # Get the user's message
    response = chatbot_response(message)  # Get the chatbot's response
    update.message.reply_text(response)  # Reply with the chatbot's response

# Main function to run the bot
def main():
    # toked recieved from BotFather
    updater = Updater(token="tokens generated by BotFather toke")
   
    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Register the /start command handler
    dp.add_handler(CommandHandler("start", start))

    # Register a message handler to handle normal text messages (non-command)
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
   
    # Start polling Telegram for new updates (messages)
    updater.start_polling()

    # Run the bot until manually stopped
    updater.idle()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()


