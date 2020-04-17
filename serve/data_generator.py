#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import os
import re
from operator import itemgetter

import pandas as pd
import pickle
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils import compute_class_weight
from tqdm.auto import tqdm
from logger import Logger
from utils import *
from api import *
import alpaca_trade_api as tradeapi


class DataGenerator:
    def __init__(self, company_code, from_date = '2020-02-01', to_date ='2020-04-08', data_path='./logs', output_path='./outputs', strategy_type='original',update=False, logger: Logger = None):
        
        self.api = tradeapi.REST('PK9SGKVYACS5TKW3B4N4', 'FY4ZWh/vMaNIgOf4LNNfOj1d79yO3Zey3MJNVJze',base_url='https://paper-api.alpaca.markets', api_version='v2')

        self.company_code = company_code
        self.from_date = from_date
        self.to_date = to_date
        self.strategy_type = strategy_type
        self.logger = logger
        self.data_path = data_path
        self.output_path = output_path
        self.start_col = 'open'
        self.end_col = 'eom_26'
        self.update = update
    
    
    def pred_generate_data(self):
        
        self.df_data = self.download_stock_data()
        #print("print 1")
        #print(self.df_data.shape)
        self.df = self.create_features()
        self.df.to_csv('pred_df_AAPL.csv',index=False)
        
        return self.df
    
    
    def log(self, text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)

    def download_stock_data(self):
        #print(self.from_date)
        #print(self.to_date)
        df = self.api.polygon.historic_agg_v2(self.company_code, 1, 'day', _from=self.from_date, to=self.to_date).df
        print(df.shape)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.date
        df['adjusted_close'] = df.close.copy()
        df = df.reset_index()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']

        #print(df.shape)
        
        
        return df
        

    def calculate_technical_indicators(self, df, col_name, intervals):
        # get_RSI(df, col_name, intervals)  # faster but non-smoothed RSI
        # get_RSI_smooth(df, col_name, intervals)  # momentum
        get_williamR(df, col_name, intervals)  # momentum
        get_mfi(df, intervals)  # momentum
        # get_MACD(df, col_name, intervals)  # momentum, ready to use +3
        # get_PPO(df, col_name, intervals)  # momentum, ready to use +1
        get_ROC(df, col_name, intervals)  # momentum
        get_CMF(df, col_name, intervals)  # momentum, volume EMA
        get_CMO(df, col_name, intervals)  # momentum
        get_SMA(df, col_name, intervals)
        get_SMA(df, 'open', intervals)
        get_EMA(df, col_name, intervals)
        get_WMA(df, col_name, intervals)
        get_HMA(df, col_name, intervals)
        get_TRIX(df, col_name, intervals)  # trend
        get_CCI(df, col_name, intervals)  # trend
        get_DPO(df, col_name, intervals)  # Trend oscillator
        get_kst(df, col_name, intervals)  # Trend
        get_DMI(df, col_name, intervals)  # trend
        get_BB_MAV(df, col_name, intervals)  # volatility
        # get_PSI(df, col_name, intervals)  # can't find formula
        get_force_index(df, intervals)  # volume
        get_kdjk_rsv(df, intervals)  # ready to use, +2*len(intervals), 2 rows
        get_EOM(df, col_name, intervals)  # volume momentum
        get_volume_delta(df)  # volume +1
        get_IBR(df)  # ready to use +1

    def create_labels(self, df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        self.log("creating label with original paper strategy")
        row_counter = 0
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        print("Calculating labels")
        pbar = tqdm(total=total_rows)

        while row_counter < total_rows:
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = (window_begin + window_end) / 2

                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1
                for i in range(window_begin, window_end + 1):
                    price = df.iloc[i][col_name]
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                if max_index == window_middle:
                    labels[row_counter] = 0
                elif min_index == window_middle:
                    labels[row_counter] = 1
                else:
                    labels[row_counter] = 2

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels

    def create_labels_price_rise(self, df, col_name):
        """
        labels data based on price rise on next day
          next_day - prev_day
        ((s - s.shift()) > 0).astype(np.int)
        """

        df["labels"] = ((df[col_name] - df[col_name].shift()) > 0).astype(np.int)
        df = df[1:]
        df.reset_index(drop=True, inplace=True)

    def create_label_mean_reversion(self, df, col_name):
        """
        strategy as described at "https://decodingmarkets.com/mean-reversion-trading-strategy"

        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels
        """

        self.log("creating labels with mean mean-reversion-trading-strategy")
        get_RSI_smooth(df, col_name, [3])  # new column 'rsi_3' added to df
        rsi_3_series = df['rsi_3']
        ibr = get_IBR(df)
        total_rows = len(df)
        labels = np.zeros(total_rows)
        labels[:] = np.nan
        count = 0
        for i, rsi_3 in enumerate(rsi_3_series):
            if rsi_3 < 15:  # buy
                count = count + 1

                if 3 <= count < 8 and ibr.iloc[i] < 0.2:  # TODO implement upto 5 BUYS
                    labels[i] = 1

                if count >= 8:
                    count == 0
            elif ibr.iloc[i] > 0.7:  # sell
                labels[i] = 0
            else:
                labels[i] = 2

        return labels

    def create_label_short_long_ma_crossover(self, df, col_name, short, long):
        """
        if short = 30 and long = 90,
        Buy when 30 day MA < 90 day MA
        Sell when 30 day MA > 90 day MA

        Label code : BUY => 1, SELL => 0, HOLD => 2

        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy

        returns : numpy array with integer codes for labels
        """

        self.log("creating label with {}_{}_ma".format(short, long))

        def detect_crossover(diff_prev, diff):
            if diff_prev >= 0 > diff:
                # buy
                return 1
            elif diff_prev <= 0 < diff:
                return 0
            else:
                return 2

        get_SMA(df, 'close', [short, long])
        labels = np.zeros((len(df)))
        labels[:] = np.nan
        diff = df['close_sma_' + str(short)] - df['close_sma_' + str(long)]
        diff_prev = diff.shift()
        df['diff_prev'] = diff_prev
        df['diff'] = diff

        res = df.apply(lambda row: detect_crossover(row['diff_prev'], row['diff']), axis=1)
        print("labels count", np.unique(res, return_counts=True))
        df.drop(columns=['diff_prev', 'diff'], inplace=True)
        return res

    def create_features(self):
        
        df = self.df_data 
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        intervals = range(6, 27)  # 21
        
        self.calculate_technical_indicators(df, 'close', intervals)
        self.log("Saving dataframe...")

        prev_len = len(df)
        
        #print(df.dropna())
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        self.logger.append_log("Dropped {0} nan rows before label calculation".format(prev_len - len(df)))
        
        select_cols= ['timestamp','open','high','low','close','volume','adjusted_close','wr_6','wr_7','wr_8','wr_9','wr_10','wr_11','wr_12','wr_13','wr_14','wr_15','wr_16','wr_17','wr_18','wr_19','wr_20','wr_21','wr_22','wr_23','wr_24','wr_25','wr_26','mfi_6','mfi_7','mfi_8','mfi_9','mfi_10','mfi_11','mfi_12','mfi_13','mfi_14','mfi_15','mfi_16','mfi_17','mfi_18','mfi_19','mfi_20','mfi_21','mfi_22','mfi_23','mfi_24','mfi_25','mfi_26','roc_6','roc_7','roc_8','roc_9','roc_10','roc_11','roc_12','roc_13','roc_14','roc_15','roc_16','roc_17','roc_18','roc_19','roc_20','roc_21','roc_22','roc_23','roc_24','roc_25','roc_26','cmf_6','cmf_7','cmf_8','cmf_9','cmf_10','cmf_11','cmf_12','cmf_13','cmf_14','cmf_15','cmf_16','cmf_17','cmf_18','cmf_19','cmf_20','cmf_21','cmf_22','cmf_23','cmf_24','cmf_25','cmf_26','cmo_6','cmo_7','cmo_8','cmo_9','cmo_10','cmo_11','cmo_12','cmo_13','cmo_14','cmo_15','cmo_16','cmo_17','cmo_18','cmo_19','cmo_20','cmo_21','cmo_22','cmo_23','cmo_24','cmo_25','cmo_26','close_sma_6','close_sma_7','close_sma_8','close_sma_9','close_sma_10','close_sma_11','close_sma_12','close_sma_13','close_sma_14','close_sma_15','close_sma_16','close_sma_17','close_sma_18','close_sma_19','close_sma_20','close_sma_21','close_sma_22','close_sma_23','close_sma_24','close_sma_25','close_sma_26','open_sma_6','open_sma_7','open_sma_8','open_sma_9','open_sma_10','open_sma_11','open_sma_12','open_sma_13','open_sma_14','open_sma_15','open_sma_16','open_sma_17','open_sma_18','open_sma_19','open_sma_20','open_sma_21','open_sma_22','open_sma_23','open_sma_24','open_sma_25','open_sma_26','ema_6','ema_7','ema_8','ema_9','ema_10','ema_11','ema_12','ema_13','ema_14','ema_15','ema_16','ema_17','ema_18','ema_19','ema_20','ema_21','ema_22','ema_23','ema_24','ema_25','ema_26','wma_6','wma_7','wma_8','wma_9','wma_10','wma_11','wma_12','wma_13','wma_14','wma_15','wma_16','wma_17','wma_18','wma_19','wma_20','wma_21','wma_22','wma_23','wma_24','wma_25','wma_26','hma_0','hma_1','hma_2','hma_3','hma_4','hma_5','hma_6','hma_7','hma_8','hma_9','hma_10','hma_11','hma_12','hma_13','hma_14','hma_15','hma_16','hma_17','hma_18','hma_19','hma_20','trix_6','trix_7','trix_8','trix_9','trix_10','trix_11','trix_12','trix_13','trix_14','trix_15','trix_16','trix_17','trix_18','trix_19','trix_20','trix_21','trix_22','trix_23','trix_24','trix_25','trix_26','cci_6','cci_7','cci_8','cci_9','cci_10','cci_11','cci_12','cci_13','cci_14','cci_15','cci_16','cci_17','cci_18','cci_19','cci_20','cci_21','cci_22','cci_23','cci_24','cci_25','cci_26','dpo_6','dpo_7','dpo_8','dpo_9','dpo_10','dpo_11','dpo_12','dpo_13','dpo_14','dpo_15','dpo_16','dpo_17','dpo_18','dpo_19','dpo_20','dpo_21','dpo_22','dpo_23','dpo_24','dpo_25','dpo_26','kst_6','kst_7','kst_8','kst_9','kst_10','kst_11','kst_12','kst_13','kst_14','kst_15','kst_16','kst_17','kst_18','kst_19','kst_20','kst_21','kst_22','kst_23','kst_24','kst_25','kst_26','dmi_6','dmi_7','dmi_8','dmi_9','dmi_10','dmi_11','dmi_12','dmi_13','dmi_14','dmi_15','dmi_16','dmi_17','dmi_18','dmi_19','dmi_20','dmi_21','dmi_22','dmi_23','dmi_24','dmi_25','dmi_26','bb_6','bb_7','bb_8','bb_9','bb_10','bb_11','bb_12','bb_13','bb_14','bb_15','bb_16','bb_17','bb_18','bb_19','bb_20','bb_21','bb_22','bb_23','bb_24','bb_25','bb_26','fi_6','fi_7','fi_8','fi_9','fi_10','fi_11','fi_12','fi_13','fi_14','fi_15','fi_16','fi_17','fi_18','fi_19','fi_20','fi_21','fi_22','fi_23','fi_24','fi_25','fi_26','rsv_6','kdjk_6','rsv_7','kdjk_7','rsv_8','kdjk_8','rsv_9','kdjk_9','rsv_10','kdjk_10','rsv_11','kdjk_11','rsv_12','kdjk_12','rsv_13','kdjk_13','rsv_14','kdjk_14','rsv_15','kdjk_15','rsv_16','kdjk_16','rsv_17','kdjk_17','rsv_18','kdjk_18','rsv_19','kdjk_19','rsv_20','kdjk_20','rsv_21','kdjk_21','rsv_22','kdjk_22','rsv_23','kdjk_23','rsv_24','kdjk_24','rsv_25','kdjk_25','rsv_26','kdjk_26','eom_6','eom_7','eom_8','eom_9','eom_10','eom_11','eom_12','eom_13','eom_14','eom_15','eom_16','eom_17','eom_18','eom_19','eom_20','eom_21','eom_22','eom_23','eom_24','eom_25','eom_26','volume_delta']
        
        
        return df[select_cols]

#         if 'labels' not in df.columns or self.update:
#             if re.match(r"\d+_\d+_ma", self.strategy_type):
#                 short = self.strategy_type.split('_')[0]
#                 long = self.strategy_type.split('_')[1]
#                 df['labels'] = self.create_label_short_long_ma_crossover(df, 'close', short, long)
#             else:
#                 df['labels'] = self.create_labels(df, 'close')

#             prev_len = len(df)
#             df.dropna(inplace=True)
#             df.reset_index(drop=True, inplace=True)
#             self.logger.append_log("Dropped {0} nan rows after label calculation".format(prev_len - len(df)))
            
#             print("inside create_features end if condition")
#             print(df.shape)
        
#         else:
#             print("labels already calculated")
            


    def feature_selection(self):
        df_batch = self.df_by_date(None, 10)
        list_features = list(df_batch.loc[:, self.start_col:self.end_col].columns)
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x_train = mm_scaler.fit_transform(df_batch.loc[:, self.start_col:self.end_col].values)
        y_train = df_batch['labels'].values
        num_features = 225  # should be a perfect square
        topk = 350
        select_k_best = SelectKBest(f_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        select_k_best.fit(x_train, y_train)
        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)

        common = list(set(selected_features_anova).intersection(selected_features_mic))
        self.log("common selected featues:" + str(len(common)) + ", " + str(common))
        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topK"'.format(len(common),
                                                                                                    num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:225])
        self.log(str(feat_idx))
        return feat_idx

    def df_by_date(self, start_date=None, years=5):
        if not start_date:
            start_date = self.df.head(1).iloc[0]["timestamp"]

        end_date = start_date + pd.offsets.DateOffset(years=years)
        df_batch = self.df[(self.df["timestamp"] >= start_date) & (self.df["timestamp"] <= end_date)]
        return df_batch

    def get_data(self, start_date=None, years=5):
        df_batch = self.df_by_date(start_date, years)
        x = df_batch.loc[:, self.start_col:self.end_col].values
        x = x[:, self.feat_idx]
        mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
        x = mm_scaler.fit_transform(x)
        dim = int(np.sqrt(x.shape[1]))
        x = reshape_as_image(x, dim, dim)
        x = np.stack((x,) * 3, axis=-1)

        y = df_batch['labels'].values
        sample_weights = self.get_sample_weights(y)
        y = self.one_hot_enc.transform(y.reshape(-1, 1))

        return x, y, df_batch, sample_weights

    def get_sample_weights(self, y):
        """
        calculate the sample weights based on class weights. Used for models with
        imbalanced data and one hot encoding prediction.

        params:
            y: class labels as integers
        """

        y = y.astype(int)  # compute_class_weight needs int labels
        class_weights = compute_class_weight('balanced', np.unique(y), y)

        print("real class weights are {}".format(class_weights), np.unique(y))
        print("value_counts", np.unique(y, return_counts=True))
        sample_weights = y.copy().astype(float)
        for i in np.unique(y):
            sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
            # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

        return sample_weights

    def get_rolling_data_next(self, start_date=None, window_size_yrs=6, cross_val_split=0.2):
        if not start_date:
            start_date = self.batch_start_date

        x_train, y_train, df_batch_train, sample_weights = self.get_data(start_date, window_size_yrs)
        train_end_date = df_batch_train.tail(1).iloc[0]["timestamp"]
        test_start_date = train_end_date + pd.offsets.DateOffset(days=1)
        test_end_date = test_start_date + pd.offsets.DateOffset(years=self.test_duration_years)
        x_test, y_test, df_batch_test, _ = self.get_data(test_start_date, self.test_duration_years)
        x_train, x_cv, y_train, y_cv, sample_weights, _ = train_test_split(x_train, y_train, sample_weights,
                                                                           train_size=1 - cross_val_split,
                                                                           test_size=cross_val_split,
                                                                           random_state=2, shuffle=True,
                                                                           stratify=y_train)
        self.logger.append_log("data generated: train duration={}-{}, test_duration={}-{}, size={}, {}, {}".format(
            self.batch_start_date, train_end_date, test_start_date, test_end_date, x_train.shape, x_cv.shape,
            x_test.shape))

        self.batch_start_date = self.batch_start_date + pd.offsets.DateOffset(years=1)
        is_last_batch = False
        if (self.df.tail(1).iloc[0]["timestamp"] - test_end_date).days < 180:  # 6 months
            is_last_batch = True
        return x_train, y_train, x_cv, y_cv, x_test, y_test, df_batch_train, df_batch_test, \
               sample_weights, is_last_batch
