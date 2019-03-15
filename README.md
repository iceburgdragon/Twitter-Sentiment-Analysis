# Twitter-Sentiment-Analysis
Author: Ruixuan (Ryan) Yu

System: MacOS High Sierra Version 10.13.6

tensorflow version: 1.5.0
numpy version: 1.13.0

Purpose: train a Recurrent Neural Network Model to predict sentiment of Twitter posts

Testing and training data below can be downloaded from http://help.sentiment140.com/for-students) <br />
training.1600000.processed.noemoticon.csv: 1.6 million Twitter posts used for training data <br />
testdata.manual.2009.06.14.csv: Twitter posts used for test data <br />

Python files
sentiment_preprocessing.py: for preparing training and testing data to be input for the RNN network. Run this first if "million_rnn_input.pickle" need to be prepared. It will create the following files:
train_set.csv: file where the tweets and their corresponding labels for the training data have been extracted
test_set.csv: file where the tweets and their corresponding labels for the testing data have been extracted
vocabulary.pickle: saved dict object that contains words taken from the train_set.csv as keys and integer IDs as values
million_rnn_input.pickle: saved training and testing features and labels to be fed into the RNN network

sentiment_rnn_model.py: for training the model. Model will be saved as "rnn_model_mil.h5"
test_rnn_model.py: uses the trained model to predict sentiment of Twitter posts. A saved model must exist before this can be ran

Twitter posts for testing model:
obama.txt: Twitter posts that mention the keyword "Obama"
bieber.txt: Twitter posts that mention the keyword "Bieber"
marvel.txt: Twitter posts that mention the keyword "Captain Marvel"
