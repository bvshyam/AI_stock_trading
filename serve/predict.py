import argparse
import json
import os
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta,date
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
    
    if content_type == 'text/plain':
    
        global trade_sticker

        trade_sticker = serialized_input_data.decode('utf-8')
        print(trade_sticker)

        print("getting input data")


        api = tradeapi.REST('PK9SGKVYACS5TKW3B4N4', 'FY4ZWh/vMaNIgOf4LNNfOj1d79yO3Zey3MJNVJze', 
                            base_url='https://paper-api.alpaca.markets', api_version='v2')


        logger = Logger('./logs', '_')

        # Set start and end date
        end_date = str(date.today())
        start_date = str(date.today() + timedelta(days=-60))


        data_gen = DataGenerator(trade_sticker,start_date,end_date,'./logs','./outputs','original', False, logger)

        pred_df = data_gen.pred_generate_data()


        print('Deserializing the input data.')

        # Load the scaler object from S3 bucket

        scaler_info_path = os.path.join("/opt/ml/model/", 'scaler.pkl')

        scaler = load(open(scaler_info_path, 'rb'))

        test_df_model_scaled = torch.tensor(scale(pred_df.drop(['timestamp'],axis=1), scaler)).float()


        return test_df_model_scaled

    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)
    
    
    
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    
    api = tradeapi.REST('PK9SGKVYACS5TKW3B4N4', 'FY4ZWh/vMaNIgOf4LNNfOj1d79yO3Zey3MJNVJze',
                            base_url='https://paper-api.alpaca.markets', api_version='v2')
    
    classes = ['buy','sell', 'hold']
    
    print("prediction_output")
    print(prediction_output)
    
    predict_classes = np.argmax(prediction_output,axis=1)
    
    print(predict_classes)
    
    predict_indicator = classes[int(predict_classes[-1])]
    
    print(predict_indicator)
    
    try:
        positions = api.list_positions()
        
        if len(positions) > 0:
            positions = positions[0]
            if positions.symbol == trade_sticker and int(positions.qty) > 0 and predict_indicator =='sell':
                api.submit_order(symbol=trade_sticker,qty=positions.qty,side=predict_indicator
                                 ,type='market',time_in_force='gtc')

            print("Sold {} of {} successfully".format(positions.qty,trade_sticker))

        elif len(positions) == 0:
            
            #Calcuate logic for quantity
            qty = 10
            if predict_indicator =='buy':

                api.submit_order(symbol=trade_sticker,qty=qty,side=predict_indicator
                             ,type='market',time_in_force='gtc')    

            print("Bought {} of {} successfully".format(qty,trade_sticker))
        else:
            print("Condition did not satisfy")
    except:
        print("Did nothing on Hold")
        pass

    print(positions)
    
    return str(positions)



def predict_fn(input_data, model):
    print('Generating Synthetic data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    input_data = input_data.to(device)
    
    print("Received input data")
    print(input_data)
    
    # Make sure to put the model into evaluation mode
    model.eval()

    result = model(input_data).detach()
    
    print(result)

    return result

