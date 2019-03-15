'''
RNN network for analyzing sentiment
'''

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, LSTM
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_confusion_matrix(model, test_x, test_y):
    '''
    Plots confusion matrix using test data

    arguments
    model: trained RNN model used for prediction
    test_x: test features
    test_y: test labels
    '''

    # create an array of predicted labels
    pred = model.predict(test_x, verbose=0)
    pred_array = []
    for y in pred:
        pred_array.append(np.argmax(y))
    # prepare an array of actual labels
    y_array = []
    for y in test_y:
        y_array.append(np.argmax(y))
    cm = metrics.confusion_matrix(y_array, pred_array)
    # plot confusion matrix
    labels = ['negative', 'positive']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j])
    plt.show()


# load the training and testing data created from sentiment_preprocessing_rnn.py
current_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_dir, 'million_rnn_input.pickle'), "rb") as input_file:
    train_X, train_Y, test_X, test_Y = pickle.load(input_file)

# create RNN model
model = Sequential()
# vocabulary size is 2345, input_length is the length of each sentiment
model.add(Embedding(2340, 32, input_length=30))
model.add(LSTM(200))
# label has 2 classes
model.add(Dense(2, activation="sigmoid"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=["categorical_accuracy"])

batch_size = 64
n_epochs = 3

# create validation dataset
valid_X, valid_Y = train_X[:batch_size], train_Y[:batch_size]
train_X2, train_Y2 = train_X[batch_size:], train_Y[batch_size:]

model.fit(train_X2, train_Y2, validation_data=(valid_X, valid_Y),
batch_size=batch_size, epochs=n_epochs)

# evaluate accurcay using test data, and print accuracy
accuracy = model.evaluate(test_X, test_Y, verbose=0)
print "Test accuracy: ", accuracy[1]
# plot confusion matrix
print "Plotting Confusion Matirx"
plot_confusion_matrix(model, test_X, test_Y)

# save model
model_path = os.path.join(current_dir, 'rnn_model_mil.h5')
model.save(model_path, include_optimizer=False)
print "RNN model saved as: ", model_path

