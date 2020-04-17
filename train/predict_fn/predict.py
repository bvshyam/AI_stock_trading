import argparse
import json
import os
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime
from api import *
import importlib
from data_generator import DataGenerator
from logger import Logger

import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from model import Net
from pickle import dump
from pickle import load


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #D = DataDiscriminator(64)
    model = Net(32)
    
    model_info = {}
    
    print("model_dir")
    print(model_dir)
    model_info_path = os.path.join(model_dir, 'model_main.pt')
    
    with open(model_info_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model



# scale train and test data to [-1, 1]
def transform_scale(train):
    # fit scaler
    
    print(len(train.columns))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    return scaler


# scale train and test data to [-1, 1]
def scale(dataset, scaler):
    # transform train
    
    
    dataset = scaler.transform(dataset)
    print(dataset.shape)
    return dataset

# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]




def input_fn(serialized_input_data, content_type):
    
    data = serialized_input_data #.decode('utf-8')
    print(data)
    
    print("getting input data")
    
    trade_sticker = data
    
    api = tradeapi.REST('PK9SGKVYACS5TKW3B4N4', 'FY4ZWh/vMaNIgOf4LNNfOj1d79yO3Zey3MJNVJze', 
                        base_url='https://paper-api.alpaca.markets', api_version='v2')
    
          
    logger = Logger('./logs', '_')

    data_gen = DataGenerator(trade_sticker,'2020-02-09','2020-03-20','./logs','./outputs','original', False, logger)
          
    pred_df = data_gen.pred_generate_data()
    
    print('Deserializing the input data.')
    
    scaler_info_path = os.path.join('scaler.pkl')
    
    scaler = load(open(scaler_info_path, 'rb'))
    
    test_df_model_scaled = torch.tensor(scale(pred_df.drop(['timestamp'],axis=1), scaler).float())
     
    print(test_df_model_scaled)
    
    
    print(content_type)
    
    return test_df_model_scaled
    
    if content_type == 'text/plain':
        
        data = serialized_input_data#.decode('utf-8')
        
        print(data)
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
    
    
    
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    
    api = tradeapi.REST('PK9SGKVYACS5TKW3B4N4', 'FY4ZWh/vMaNIgOf4LNNfOj1d79yO3Zey3MJNVJze',
                            base_url='https://paper-api.alpaca.markets', api_version='v2')
    
    classes = ['buy','sell', 'hold']
    
    
    trade_sticker = 'AAPL'
    
    predict_classes = np.argmax(prediction_output,axis=1)
    
    predict_indicator = classes[int(predict_classes)]
    
    print(predict_indicator)
    
    positions = api.list_positions()[0]
    
    if positions.symbol == trade_sticker and int(positions.qty) > 0:
        api.submit_order(symbol=trade_sticker,qty=positions.qty,side=predict_indicator
                         ,type='market',time_in_force='gtc')
        print("Sold {} of {} successfully".format(positions.qty,trade_sticker))
    elif positions.symbol != trade_sticker:
        api.submit_order(symbol=trade_sticker,qty=positions.qty,side=predict_indicator
                         ,type='market',time_in_force='gtc')    

        print("Bought {} of {} successfully".format(positions.qty,trade_sticker))
    else:
        print("Did nothing on Hold")
        pass
    
    print(positions)
    
    positions = api.list_positions()[0]
    
    return str(positions)



def predict_fn(input_data, model):
    print('Generating Synthetic data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    input_data = input_data.to(device)
    
    print("Received input data")
    print(input_data)
    
    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    
    # data = input_noise
    # data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0
    
    result = model(input_data).detach()

    return result

