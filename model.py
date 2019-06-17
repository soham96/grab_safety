import argparse
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Conv1D, Dense, TimeDistributed, ConvLSTM2D, Dropout, LSTM, MaxPooling1D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve

from utils.helpers import read_data
from preprocess import *


def get_results(model, model_type):
    '''
    Function to get the result from testing the model.
    Args:
        model: The trained model
        model_type: The type of model. For instance 'LSTM'
    Returns:
        NA. Prints the statistics of the results of training the model
    '''
    x, y = get_test_batch(batches=1, model = model_type) 
    predicted_class= model.predict(x, verbose=1)

    predicted_class = [np.argmax(r) for r in predicted_class]
    test_y = [np.argmax(r) for r in y]

    print('Confusion matrix is \n', confusion_matrix(test_y, predicted_class))
    print('tn, fp, fn, tp =')
    print(confusion_matrix(test_y, predicted_class).ravel())
    # Precision
    print('Precision = ', precision_score(test_y, predicted_class))
    # Recall
    print('Recall = ', recall_score(test_y, predicted_class))
    # f1_score
    print('f1_score = ', f1_score(test_y, predicted_class))
    # cohen_kappa_score
    print('cohen_kappa_score = ', cohen_kappa_score(test_y, predicted_class))
    # roc_auc_score
    print('roc_auc_score = ', roc_auc_score(test_y, predicted_class))

def LSTM_train(args):
    '''
    Function to train a Vanilla LSTM model
    '''
    data_df, label_df=read_data()
    
    model = Sequential()
    model.add(LSTM(512, input_shape=(1200,11)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit_generator(prepare_data_for_training(data_df, label_df, seq_len=1200), steps_per_epoch=1000, epochs=int(args.epochs), verbose=1)

    get_results(model=model, model_type='LSTM')

    if args.save_to:
        print("Saving Model")
        model.save(args.save_to)

def CNN_LSTM(args):
    '''
    Function to train a CNN LSTM
    '''
    data_df, label_df=read_data()
    
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'), input_shape=(None,120,11)))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(prepare_data_for_training(data_df, label_df, seq_len=1200, model_type='CNN_LSTM', batch_size=1), steps_per_epoch=1000, epochs=int(args.epochs), verbose=1)
    
    get_results(model=model, model_type='CNN_LSTM')

    if args.save_to:
        print("Saving Model")
        model.save(args.save_to)

def Conv_LSTM(args):
    '''
    Function to train a Convolutional LSTM
    '''
    data_df, label_df=read_data()
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(10, 1, 120, 11)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(prepare_data_for_training(data_df, label_df, seq_len=1200, model_type='CONV_LSTM', batch_size=1), steps_per_epoch=1000, epochs=int(args.epochs), verbose=1)

    get_results(model=model, model_type='CONV_LSTM')

    if args.save_to:
        print("Saving Model")
        model.save(args.save_to)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--model', default='LSTM', help='There are three models you can train: LSTM, CNN_LSTM and CONV_LSTM.\
           Choose one of those values. Default: LSTM')
    a.add_argument('--save_to', help='Location to save your model to.')
    a.add_argument('--epochs', help='Number of epochs to train your model for. Default = 1', default=1)

    args = a.parse_args()
    if args.model == 'LSTM':
        LSTM_train(args)
    elif args.model == 'CNN_LSTM':
        CNN_LSTM(args)
    elif args.model == 'CONV_LSTM':
        Conv_LSTM(args)
    else:
        raise ValueError("Please specify model using the --model tag. Use one of:LSTM, CNN_LSTM and CONV_LSTM. see --help for more info.")
