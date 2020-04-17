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


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Net(32)
    
    model_info = {}
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

def _get_train_data_loader(batch_size, train_data):
    print("Get train data loader.")
    
    train_X = torch.from_numpy(train_data.drop(['labels'],axis=1).values).float()
    
    train_Y = torch.from_numpy(train_data['labels'].values).float()
    
    train_ds = torch.utils.data.TensorDataset(train_X,train_Y)

    return torch.utils.data.DataLoader(train_ds,shuffle=False, batch_size=batch_size)



def train(model, scaler, train_loader, epochs, optimizer, criterion, train_on_gpu):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    scaler       - Scaler object for MinMax.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training.
    train_on_gpu - Where the model and data should be loaded (gpu or cpu).
    """
    
    
    print_every=50
    losses = []
    
    if train_on_gpu:
        model.cuda()
    
    # train the network
    for epoch in range(epochs):

        for batch_i, (train_x,train_y) in enumerate(train_loader):
            
            #Get batch size
            batch_size = train_x.size()[0]
            
            #Scale the data
            train_x = torch.tensor(scale(train_x, scaler))
            
            #print(train_x[:10])
            
            train_y = train_y.long()
            
            model.train()
            
            model.zero_grad()
            
            # In GPU
            if train_on_gpu:
                train_x = train_x.float().cuda()
                train_y = train_y.cuda()
                
            output = model(train_x)
            
            loss = criterion(output,train_y)
            
            loss.backward()
            optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append losses
                losses.append((loss.item()))
                # print  losses
                print('Epoch [{:5d}/{:5d}] | loss: {:6.4f}'.format(
                        epoch+1, epochs, loss.item()))
        
    return model



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ticker', type=str, default='AAPL', metavar='S',
                        help='random seed (default: 1)')
        
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

       # Model Parameters
    parser.add_argument('--hidden_dim', type=int, default=32, metavar='N',
                        help='size of the Hidden dim (default: 64)')
    
        # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus',type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = parser.parse_args()
    
    device = torch.cuda.is_available()
    print("Using device {}.".format(device))
    
    logger = Logger('./logs', '_')
    
    #Change the dates accordingly for training
    data_gen = DataGenerator(args.ticker,'2010-01-09','2020-03-20','./logs','./outputs','original', False, logger)
    
    df = data_gen.generate_data()
    
    df.drop(['timestamp'],axis=1,inplace = True)    

    scaler = transform_scale(df.drop(['labels'],axis=1))

    train_loader = _get_train_data_loader(args.batch_size, df)

    model = Net(args.hidden_dim)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

    model = train(model, scaler, train_loader, args.epochs, optimizer, criterion, device)
    
    print(args.model_dir)
    
    model_path = os.path.join(args.model_dir, 'model_main.pt')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
        
    dump(scaler, open(os.path.join(args.model_dir,'scaler.pkl'), 'wb'))

    
    print(model)
    
    print("Model generated successfully")
    
    
    
    
    
    