import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Conv1D, Dense, TimeDistributed, ConvLSTM2D, Dropout, LSTM, MaxPooling1D
from keras.utils import to_categorical

from preprocess import *

data_df, label_df=read_data()

def LSTM_train():
    model = Sequential()
    model.add(LSTM(100, input_shape=(1200,11)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit_generator(prepare_data_for_training(data_df, label_df, seq_len=1200), steps_per_epoch=1000, epochs=5, verbose=1)

def CNN_LSTM():
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,120,11)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(prepare_data_for_training(data_df, label_df, seq_len=1200, model_type='CNN_LSTM', batch_size=1), steps_per_epoch=1000, epochs=5, verbose=1)

def Conv_LSTM():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

if __name__=='__main__':
    CNN_LSTM()
    Conv_LSTM()
