'''
This program uses the trained RNN model to analyze
the sentiment of new data from twitter

Twitter data is collected using the Twitter API, and 
stored in a txt file
'''

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
import pickle
import numpy as np
import json
from nltk.tokenize import word_tokenize


def load_data(file):
    '''
    Takes tweets from collected data and append
    them to an array of strings

    arguments
    file: full path of collected twitter data

    returns
    tweet_text: array of strings
    '''

    # to store tweets as strings
    tweet_text = []
   
    with open(file, 'r') as f:
        for line in f:
            try:
                raw_tweet = json.loads(line)
                #extract only tweets in English and append
                if raw_tweet.get('lang') == 'en':
                    tweet = raw_tweet.get('text')
                    tweet = tweet.decode('utf-8')
                    tweet_text.append(tweet)
            except:
                continue
    
    return tweet_text

def create_features(texts, vocab, padding=30):
    '''
    create feature vectors of integer ID

    arguments
    texts: array of tweet strings
    vocab: full path of the vocabulary dict used to
        convert words into integer IDs
    padding: length of the feature vector

    returns
    padded_features: feature vector to be input for the
        trained RNN network
    '''

    # load vocabulary
    with open(vocab, 'rb') as f:
        vocabulary = pickle.load(f)
    # array for appending features and labels
    features = []

    for tweet in texts:
        words = word_tokenize(tweet.lower())
        # convert each word to integer ID
        # using the vocabulary dict
        int_vec = []
        for w in words:
            if w in vocabulary:
                int_vec.append(vocabulary[w])
        # append features
        features.append(int_vec)

    # pad the array
    padded_features = pad_sequences(features, maxlen=padding)
    return padded_features

# path to current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
# path to vocabulary dict
vocab_file = os.path.join(current_dir, 'vocabulary.pickle')

# Indicate which twitter data will be used for the analysis below
# Feature vectors will be created from each tweet
tweets_data = load_data(os.path.join(current_dir, 'bieber.txt'))
X = create_features(tweets_data, vocab_file)

# Load the trained RNN model
loaded_model = load_model(os.path.join(current_dir,'rnn_model_mil.h5'))
print("Loaded RNN model")

print("Analyzing sentiments for {} tweets").format(len(X))
analysis = loaded_model.predict(X, verbose=0)

pos_sentiments = sum([np.argmax(a) for a in analysis])/float(len(analysis))
print "Percentage of positive sentiments is {0:.1f}%.".format(pos_sentiments*100)

for i in range(10):
    print tweets_data[i]
    print analysis[i]
    if np.argmax(analysis[i]) == 1:
        print "positive"
    else:
        print "negative"




