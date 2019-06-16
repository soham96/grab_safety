import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical

from preprocess import *

data_df, label_df=read_data()

model = Sequential()
model.add(LSTM(100, input_shape=(1200,11)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit_generator(prepare_data_for_training(data_df, label_df, seq_len=1200), steps_per_epoch=1000, epochs=5, verbose=1)
