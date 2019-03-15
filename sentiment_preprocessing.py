'''
This preprocessing program takes sentiment data from sentiment140
twitter data and extracts features and labels
First a vocabulary is created using words from the sentiment files.

Features have shape = (number of sentiments, length of sentiment (default=50))
feature is an array of integers where each integer represents a word
'''

from nltk.tokenize import word_tokenize
import numpy as np 
import random
import pickle
import os
from io import open
import operator
from keras.preprocessing.sequence import pad_sequences


def process_file(fin, fout):
    '''
    Takes the raw csv file as input and saves a new csv
    file containing just the tweet and its corresponding
    sentiment label

    arguments
    fin: full path of the input file
    fout: full path of the output file
    '''

    # to hold the extracted tweets and sentiments
    data = []
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                # read each line and determine initial polarity
                # [0,1] for positive sentiment
                # [1,0] for negative sentiment
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0]
                elif initial_polarity == '4':
                    initial_polarity = [0,1]

                tweet = line.split(',')[-1]
                outline = str(initial_polarity)+':::'+tweet
                # append extracted tweet and corresponding label
                data.append(outline)
        except Exception as e:
            print(str(e))
    
    # shuffle the data
    random.shuffle(data)
    # write each string in the array to a new file
    with open(fout, 'a') as f:
        for s in data:
            f.write(s)
    

def save_vocabulary(fin, fout, min_count=0, max_count=50000):
    '''
    takes a csv file, and outputs a vocabulary with each word
    paired with its integer ID

    arguments
    fin: full path of the input file
    fout: full path of the output file
    min_count: words that occur less than this value will be
        excluded from the vocabulary
    max_count: words that occur more than this value will be
        excluded from the vocabulary
    '''

    # dict for holding all unique words found and their counts
    word_count = {}

    with open(fin, 'r', buffering=10000, encoding='latin-1') as f:
        try:
            for i, line in enumerate(f):
                # for better speed, take every 100 tweet
                if i % 100 == 0:
                    tweet = line.split(':::')[1]
                    words = word_tokenize(tweet.lower())
                    # if new word is found, add to word_count dict
                    # otherwise, increment the count of the word by 1
                    for w in words:
                        if w not in word_count:
                            if len(w) > 1:
                                word_count[w] = 1
                        else:
                            word_count[w] += 1
        except Exception as e:
            print(str(e))
    
    print "number of unique words found: ", len(word_count)
    # sort in order of descending word count
    sorted_word_counts = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    print "Top 50 words are: ", sorted_word_counts[:50]

    # dict to hold the vocabulary with the word as the key
    # and the integer ID as the value
    vocabulary = {}
    # integer ID starts as 1
    word_id = 1
    # add each word with a corresponding integer ID to vocabulary
    # by using an array sorted by word count, words that
    # appear more frequently will have a lower integer ID,
    # which appears to improve accuracy when training
    for wc in word_count:
        # only include words with more than 1 character
        # check if its count is between min_count and max_count
        if len(wc) > 1:
            if max_count > word_count[wc] > min_count:
                vocabulary[wc] = word_id
                word_id += 1
    # save vocabulary
    with open(fout, 'wb') as f:
        pickle.dump(vocabulary, f)
    print "Saved vocabulary of size: ", len(vocabulary)


def extract_features(file, vocab, num_lines=None):
    '''
    Takes the previously shuffled data set,
    and creates a featureset of integer ID vectors

    arguments
    file: full path of the data set
    vocab: full path of the saved vocabulary
    num_lines (optional): number of lines (tweets) that will
        be used to create the featureset
        If no value is passed, the entire dataset will be used
    
    returns
    featureset: array containing arrays of integer encoded
        sentiments, and its corresponding one hot label
    '''

    # load vocabulary
    with open(vocab, 'rb') as f:
        vocabulary = pickle.load(f)
    # array for appending features and labels
    featureset = []

    with open(file, buffering=200000, encoding='latin-1') as f:
        try:
            if num_lines:
                for i in range(num_lines):
                    line = f.readline()
                    # extract label
                    label = eval(line.split(':::')[0])
                    features = []
                    # extract tweet and tokenize
                    tweet = line.split(':::')[1]
                    words = word_tokenize(tweet.lower())
                    # convert each word to integer ID
                    # using the vocabulary dict
                    for w in words:
                        if w in vocabulary:
                            features.append(vocabulary[w])
                    # append features along with its labels
                    featureset.append([features, label])
            # if no argument is given for num_lines
            # it will read through every tweet
            else:
                for line in f:
                    label = eval(line.split(':::')[0])
                    features = []
                    tweet = line.split(':::')[1]
                    words = word_tokenize(tweet)
                    for w in words:
                        if w in vocabulary:
                            features.append(vocabulary[w])
                    featureset.append([features, label])
        except Exception as e:
            print(str(e))
    
    return featureset
            
 
def create_features_labels(file, vocab, num_lines=None, padding=30):
    '''
    creates x and y sets to be fed into the network

    arguments
    featureset: output from extract_features
    padding: maximum length of the feature vector
        feature vectors that are too short will be
        padded with zeros in front
        vectors that are too long will be truncated
    
    returns
    X: feature vector of integer ID
    Y: one hot label
    '''

    featureset = extract_features(file=file, vocab=vocab, num_lines=num_lines)

    # separate the features and the labels into
    # individual array
    X = [i[0] for i in featureset]
    Y = []
    for e in featureset:
        if e[1] == [1,0]:
            Y.append([1,0])
        else:
            Y.append([0,1])
    Y2 = np.array(Y)
 
    # pad the sequence to specified length
    X = pad_sequences(X, maxlen=padding)
    
    return X, Y2


current_dir = os.path.dirname(os.path.realpath(__file__))
# name of the raw training file
train_file = os.path.join(current_dir, 'training.1600000.processed.noemoticon.csv')
# name of the raw testing file
test_file = os.path.join(current_dir, 'testdata.manual.2009.06.14.csv')

# name of the preprocessed training data
train_set_file = os.path.join(current_dir, 'train_set.csv')
# name of the preprocess test data
test_set_file = os.path.join(current_dir, 'test_set.csv')
# name of the vocabulary file
vocab_file = os.path.join(current_dir, 'vocabulary.pickle')

'''
# The code below only need to be run once
# It saves a training_set csv file and vocabulary dict pickle file
# extract relevant information from raw data and save file
process_file(train_file, train_set_file)
print("Saved extracted training set file")
# build a vocabulary and save
save_vocabulary(train_set_file, vocab_file, min_count=10, max_count=5000)
print("Saved vocabulary file")
'''

# create training data
train_x, train_y = create_features_labels(train_set_file, vocab_file, num_lines=1000000)
print("Created training set")
# create test data
process_file(test_file, test_set_file)
test_x, test_y = create_features_labels(test_set_file, vocab_file)
print("Created testing set")

# save training and test data
with open(os.path.join(current_dir, 'million_rnn_input.pickle'),'wb') as f:
    pickle.dump([train_x, train_y, test_x, test_y], f)
    print "Featuresets saved to: ", current_dir
