import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# THIS IS THE OLD VERSION OF THE PROJECT
# THAT IS NOT COMPATIBLE WITH THE WEB BROWSER
# THIS VERSION WILL RUN IN THE TERMINAL


# create a lemmatizer
lemmatizer = WordNetLemmatizer()
# load the intents file
intents = json.loads(open('intents.json').read())

# load the words, classes, and model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

# cleaning sentence function
def clean_up_sentence(sentence):
    # split the sentence by spaces
    sentence_words = nltk.word_tokenize(sentence)
    # lemmatize the each word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# convert the sentence into a bag of words
# a list full of 0's and 1's that indicate if
# the word is present
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# predecit funciton based on a sentence
def predict_class(sentence):
    # bow is bag of words
    # we get the bag of words from our bag of words function
    bow = bag_of_words(sentence)
    # result
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHHOLD = 0.25
    # if the res is larger than threshold set it to result
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHHOLD]

    # sorted with lambda key in reverse order
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list 

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print('Welcome. I am your chatbot today, Steven.')
# main function
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)