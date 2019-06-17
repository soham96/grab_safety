import glob
import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.utils import to_categorical
from sklearn.preprocessing import RobustScaler
from utils.helpers import *

def data_engineering(df):
    '''
    Function to perform data engineering.
    1) Gets the mean of acceleration and gyro for all axis's
    2) Gets RMS value for acceleration
    
    Args:
        df: DataFrame to perform data engineering on
    Returns:
        DataFrame with data engineering done
    '''
    #centering all values to mean
    for i in ['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y',\
       'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']:
        df[i]=df[i]-df[i].mean()

    #RMS Acceleration value and add to dataframe
    df["acceleration"] = (df["acceleration_x"]**2 + df["acceleration_y"]**2 + df["acceleration_z"]**2)**(1/2)
    
    return df

def prepare_data_for_training(data_df, label_df, seq_len=3000, batch_size=32, model_type='LSTM'):
    '''
    Generator that returns batches of data for trainining models
    Args:
        data_df: DataFrame containing the training data
        label_df: The lable for the data contained in data_df
        seq_len: The seq_len for one datapoint in data_df. Default: 3000
        batch_size: The batch size of data to yield per iteration. Default: 32
    Returns:
        A generator that yields batches of data in the form (X, one hot encoded label)
    '''
    bookingIDs=data_df['bookingID'].drop_duplicates().values
    while True:
        final_data=[]
        label_list=[]
        i=0
        #label_df=label_df.set_index('bookingID')
        for bookingID in bookingIDs:
            try:
                if not label_df[label_df['bookingID']==bookingID].empty:
                    temp_df=data_df[data_df['bookingID']==bookingID]
                    temp_df=temp_df.sort_values(by='second')
                    
                    temp_df=fix_length(temp_df, seq_len)
                    temp_df=data_engineering(temp_df)
                    label_list.append(int(label_df[label_df['bookingID']==bookingID]['label']))
                    temp_df=temp_df.drop(['bookingID'], axis=1)
                    final_data.append(temp_df.values)
                    i=i+1
                    if i==batch_size:
                        if 1 in label_list and 0 in label_list:
                            if model_type=='LSTM':
                                yield np.asarray(final_data), to_categorical(label_list)
                            elif model_type=='CNN_LSTM':
                                yield np.asarray(final_data).reshape((int(len(np.asarray(final_data))), 10, -1, 11)), to_categorical(label_list)
                            elif model_type=='CONV_LSTM':
                                yield np.asarray(final_data).reshape((int(len(np.asarray(final_data))), 10, 1, -1, 11)), to_categorical(label_list)
                            final_data=[]
                            label_list=[]
                            i=0
                        else:
                            i=0
            except Exception as e:
                print(e)

def get_test_batch(batches, model):
    '''
    Returns batches of data to test the model on.
    Args:
        batches: Number of batches to yield
        model: The type of model you are using. For instance 'LSTM'.
    Returns:
        x, y: The testing data x and its class y
    '''
    data_df, label_df = read_data()
    x, y= next(prepare_data_for_training(data_df, label_df, seq_len=1200, batch_size=batches, model_type=model))

    return x, y

if __name__=='__main__':
    #data_file_list=get_files('features')
    #data_df=get_df(data_file_list)

    ## In case of multiple label files, use the following two lines
    label_file_list=get_files('safety/labels')
    label_df=get_df(label_file_list)
    #
    #with open('data_df.pkl', 'wb') as f:
    #    pickle.dump(data_df, f)
    with open('data_df.pkl', 'rb') as f:
        data_df=pickle.load(f)

    final_data_df, label_list=prepare_data_for_training(data_df, label_df) 
