import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

# first we create a word lemmatizers
# this lemmatizes all of the words so that the chat bot can recognize
# words within sentences and of different cases
lemmatizer = WordNetLemmatizer()

# load json file
# read the contents of the file as text
# pass that text to the loads funciton
# creates a json object (dictionary) from our 'intents.json' file
intents = json.loads(open('intents.json').read())

words = []
classes =[]

# the combination of words and intents
documents = []
# these are the letters to be ignored
ignore_letters = ['?', '!', '.', ',']

# for loop to access each intent
# each intent has certain sub values
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # word_tokenize is getting a text and splitting it into individual words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # used to check that the words belong to a certain tag
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize each word in the list if it is not a in 'ingore_letters'
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# set will eliminate all duplicates
# sorted will turn in back into a sortedlist 
words = sorted(set(words))

classes = sorted(set(classes))

# pickle file to write words and classes into a binary file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# set word values to 0 or 1 if it is occuring
training = []
# make as many 0's as there are classes
output_empty = [0] * len(classes)

# this loop iterates through the documents in order
# to create a training list called "training"
# this training list will be used to train the neural network
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # check if each word occurs in a pattern
    # values of 1 if the word occurs, and 0 if it does not
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# shuffle the training data
random.shuffle(training)
training = np.array(training)

# split into x and y values
# these are the features and the labels used to train the neural network
train_x = training[:, :len(words)]
train_y = training[:, len(words):]


# build the neural network
model = tf.keras.Sequential()
# first layer is an input layer, Dense
# has 128 neurons with an input shape that depends on x training data
# activation funciton is a rectified linear unit "relu"
model.add(tf.keras.layers.Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
# add dropout to prevent over fitting
model.add(tf.keras.layers.Dropout(0.5))
# add another dense layer with 64 neurons with relu activation function
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
# same number of neurons as there are classes (train_y[0])
# activation funciton is softmax
# softmax will sum up / scale the results in the output layer
# so that they all add up to one
model.add(tf.keras.layers.Dense(len(train_y[0]), activation = 'softmax'))

# set learning rate
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# input training data (x) and output training data (y)
# feeds the same data 200 times into the neural netowrk (epochs = 200)
# verbose is set to 1 so we get a medium amount of information
model.fit(train_x, train_y, epochs = 200, batch_size = 5, verbose = 1)
# save the model
model.save('chatbot_model.model')
print('Done')



