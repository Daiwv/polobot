# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:30:36 2017

@author: xangma
Code structure is as follows:

Settings
Imports
Functions
Preprocessing
NN settings
NN running
Plotting and other random stuff

# To do: 
- Make only shuffle switch just for training data.
- Change price_dir_only to be a 3 class problem instead of 2 like it is currently. This is so it can predict 0 price swing.
- Create backtest function to simulate trades+ trading fees.

"""

from __future__ import print_function

# SETTINGS

# features to choose from:
# array(['close', 'high', 'low', 'open', 'quoteVolume', 'volume', 'weightedAverage', 'sma', 'bbtop', 'bbbottom', 'bbrange', 'bbpercent', 'emaslow', 'emafast', 'macd', 'rsi_24', 'bodysize', 'shadowsize', 'percentChange']
onlyuse = ['bbrange', 'bbpercent','rsi_30','rsi_24','rsi_12','rsi_8','macd']

test_size = 0.2
shuffle_cats = False # maybe deprecated, check
n_cat=20000
modelname='polo_btc_eth'
load_old_model = False
run_training = True
run_pred= True
batch_size = 20000
epochs = 5000

price_dir_only = True
generate_features = False
makedata_convtest= True
nb_ticks_history = 20
shuffle_whole_cat = True
shuffle_training = True

# IMPORTS 
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, AveragePooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import Model
from keras.layers import Input
import threading
from poloniex import Poloniex
polo = Poloniex()

import websocket # pip install websocket-client
from multiprocessing.dummy import Process as Thread
import json
import logging
from time import time,sleep
from datetime import datetime
import pprint
from operator import itemgetter
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)

# FUNCTIONS
def rsi(df, window, targetcol='weightedAverage', colname='rsi'):
    """ Calculates the Relative Strength Index (RSI) from a pandas dataframe
    http://stackoverflow.com/a/32346692/3389859
    """
    colname = colname+'_%i'%window
    series = df[targetcol]
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[window - 1]] = np.mean(u[:window])
    u = u.drop(u.index[:(window - 1)])
    # first value is sum of avg losses
    d[d.index[window - 1]] = np.mean(d[:window])
    d = d.drop(d.index[:(window - 1)])
    rs = u.ewm(com=window - 1,
               ignore_na=False,
               min_periods=0,
               adjust=False).mean() / d.ewm(com=window - 1,
                                            ignore_na=False,
                                            min_periods=0,
                                            adjust=False).mean()
    df[colname] = 100 - 100 / (1 + rs)
    return df


def sma(df, window, targetcol='weightedAverage', colname='sma'):
    """ Calculates Simple Moving Average on a 'targetcol' in a pandas dataframe
    """
    df[colname] = df[targetcol].rolling(window=window, center=False).mean()
    return df


def ema(df, window, targetcol='weightedAverage', colname='ema', **kwargs):
    """ Calculates Expodential Moving Average on a 'targetcol' in a pandas
    dataframe """
    df[colname] = df[targetcol].ewm(
        span=window,
        min_periods=kwargs.get('min_periods', 1),
        adjust=kwargs.get('adjust', True),
        ignore_na=kwargs.get('ignore_na', False)
    ).mean()
    return df


def macd(df, fastcol='emafast', slowcol='emaslow', colname='macd'):
    """ Calculates the differance between 'fastcol' and 'slowcol' in a pandas
    dataframe """
    df[colname] = df[fastcol] - df[slowcol]
    return df


def bbands(df, window, targetcol='weightedAverage', stddev=2.0):
    """ Calculates Bollinger Bands for 'targetcol' of a pandas dataframe """
    if not 'sma' in df:
        df = sma(df, window, targetcol)
    df['bbtop'] = df['sma'] + stddev * df[targetcol].rolling(
        min_periods=window,
        window=window,
        center=False).std()
    df['bbbottom'] = df['sma'] - stddev * df[targetcol].rolling(
        min_periods=window,
        window=window,
        center=False).std()
    df['bbrange'] = df['bbtop'] - df['bbbottom']
    df['bbpercent'] = ((df[targetcol] - df['bbbottom']) / df['bbrange']) - 0.5
    return df


class Chart(object):
    """ Saves and retrieves chart data to/from mongodb. It saves the chart
    based on candle size, and when called, it will automaticly update chart
    data if needed using the timestamp of the newest candle to determine how
    much data needs to be updated """

    def __init__(self, api, pair, **kwargs):
        """
        api = poloniex api object
        pair = market pair
        period = time period of candles (default: 5 Min)
        """
        self.pair = pair
        self.api = api
        self.period = kwargs.get('period', self.api.MINUTE * 5)
        self.db = MongoClient()['poloniex']['%s_%s_chart' %
                                            (self.pair, str(self.period))]

    def __call__(self, size=0):
        """ Returns raw data from the db, updates the db if needed """
        # get old data from db
        old = sorted(list(self.db.find()), key=itemgetter('_id'))
        try:
            # get last candle
            last = old[-1]
        except:
            # no candle found, db collection is empty
            last = False
        # no entrys found, get last year of data to fill the db
        if not last:
            logger.warning('%s collection is empty!',
                           '%s_%s_chart' % (self.pair, str(self.period)))
            new = self.api.returnChartData(self.pair,
                                           period=self.period,
                                           start=time() - self.api.YEAR)
        # we have data in db already
        else:
            new = self.api.returnChartData(self.pair,
                                           period=self.period,
                                           start=int(last['_id']))
        # add new candles
        updateSize = len(new)
        logger.info('Updating %s with %s new entrys!',
                    self.pair + '-' + str(self.period), str(updateSize))
        # show progress
        for i in range(updateSize):
            print("\r%s/%s" % (str(i + 1), str(updateSize)), end=" complete ")
            date = new[i]['date']
            del new[i]['date']
            self.db.update_one({'_id': date}, {"$set": new[i]}, upsert=True)
        print('')
        logger.debug('Getting chart data from db')
        # return data from db
        return sorted(list(self.db.find()), key=itemgetter('_id'))[-size:]

    def dataFrame(self, size=0, window=120):
        # get data from db
        data = self.__call__(size)
        # make dataframe
        df = pd.DataFrame(data)
        # format dates
        df['date'] = [pd.to_datetime(c['_id'], unit='s') for c in data]
        # del '_id'
        del df['_id']
        # set 'date' col as index
        df.set_index('date', inplace=True)
        # calculate/add sma and bbands
        df = bbands(df, window)
        # add slow ema
        df = ema(df, window // 2, colname='emaslow')
        # add fast ema
        df = ema(df, window // 4, colname='emafast')
        # add macd
        df = macd(df)
        # add rsi
        df = rsi(df, window // 4)
        df = rsi(df, window // 5)
        df = rsi(df, window // 10)
        df = rsi(df, window // 15)
        df = rsi(df, window // 20)
        # add candle body and shadow size
        df['bodysize'] = df['open'] - df['close']
        df['shadowsize'] = df['high'] - df['low']
        # add percent change
        df['percentChange'] = df['close'].pct_change()
        return df


class dictTicker(object):

    def __init__(self, api=None):
        self.api = api
        if not self.api:
            self.api = Poloniex(jsonNums=float)
        self.tick = {}

        iniTick = self.api.returnTicker()
        self._ids = {market: iniTick[market]['id'] for market in iniTick}
        for market in iniTick:
            self.tick[self._ids[market]] = iniTick[market]

        self._ws = websocket.WebSocketApp("wss://api2.poloniex.com/",
                                          on_open=self.on_open,
                                          on_message=self.on_message,
                                          on_error=self.on_error,
                                          on_close=self.on_close)

    def on_message(self, ws, message):
        message = json.loads(message)
        if 'error' in message:
            return logger.error(message['error'])

        if message[0] == 1002:
            if message[1] == 1:
                return logger.info('Subscribed to ticker')

            if message[1] == 0:
                return logger.info('Unsubscribed to ticker')

            data = message[2]
            data = [float(dat) for dat in data]
            self.tick[data[0]] = {'id': data[0],
                                  'last': data[1],
                                  'lowestAsk': data[2],
                                  'highestBid': data[3],
                                  'percentChange': data[4],
                                  'baseVolume': data[5],
                                  'quoteVolume': data[6],
                                  'isFrozen': data[7],
                                  'high24hr': data[8],
                                  'low24hr': data[9]
                                  }

    def on_error(self, ws, error):
        logger.error(error)

    def on_close(self, ws):
        if self._t._running:
            try:
                self.stop()
            except Exception as e:
                logger.exception(e)
            try:
                self.start()
            except Exception as e:
                logger.exception(e)
                self.stop()
        else:
            logger.info("Websocket closed!")

    def on_open(self, ws):
        self._ws.send(json.dumps({'command': 'subscribe', 'channel': 1002}))

    @property
    def status(self):
        """
        Returns True if the websocket is running, False if not
        """
        try:
            return self._t._running
        except:
            return False

    def start(self):
        """ Run the websocket in a thread """
        self._t = Thread(target=self._ws.run_forever)
        self._t.daemon = True
        self._t._running = True
        self._t.start()
        logger.info('Websocket thread started')

    def stop(self):
        """ Stop/join the websocket thread """
        self._t._running = False
        self._ws.close()
        self._t.join()
        logger.info('Websocket thread stopped/joined')

    def __call__(self, market=None):
        """ returns ticker from mongodb """
        if market:
            return self.tick[self._ids[market]]
        return self.tick

logging.basicConfig(level=logging.DEBUG)
## websocket.enableTrace(True)
#ticker = dictTicker()
#try:
#    ticker.start() # Start listening to market tickers
##    for i in range(3):
##        sleep(5)
##        pprint.pprint(ticker('USDT_BTC'))
#except Exception as e:
#    logger.exception(e)
##ticker.stop()

logging.getLogger("poloniex").setLevel(logging.INFO)
logging.getLogger('requests').setLevel(logging.ERROR)
api = Poloniex(jsonNums=float)

def updateChart(market): # updates given market when called
    df = Chart(api, market).dataFrame()
    df.dropna(inplace=True)
    return df

def dropChart(market):
    df.drop(df.index, inplace=True)

def get_function(function_string): # Used to set machine learning algorithm and settings
    import importlib
    module, function = function_string.rsplit('.', 1)
    module = importlib.import_module(module)
    function = getattr(module, function)
    return function

def zcmn_scaling(array,means,stds):
    for i in range(len(means)):
        if makedata_convtest == True:
            array[:,:,i]-=means[i]
            array[:,:,i]/=stds[i]
        else:
            array[:,i]-=means[i]
            array[:,i]/=stds[i]
    return array

df = updateChart('BTC_ETH')

#ts = (df.index[-1] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
#latest_time=datetime.utcfromtimestamp(ts)

features_all = df.columns.values.astype('str') # get column names from dataframe
XX = df.values # Get column values from dataframe
XX = np.flipud(XX) # makes latest 5 minute tick XX[0] instead of XX[-1] (flips the array so most recent is at top)
percentchange = XX[:,-1] # Takes percentage train as variable to predict
XX = XX[1:,:-1] # Takes off latest tick to compensate for the shift, as we're going to shift the 'percent change since last tick' down. It will become the 'percent change for the next tick'. Also removes the percentage change from the features list 

yy=percentchange[:-1] # Shift percentage change down
# CUT DOWN FEATURE COLUMNS HERE
onlyusemask= np.array([x in onlyuse for x in features_all]) # Filters down features to those only in onlyuse array
XX = XX.T[onlyusemask[:-1]] # Need to do a stupid transpose
XX = XX.T
features = features_all[onlyusemask] # Filter feature labels
# FEATURE GENERATION. Take feature values from last tick
XX_generated = XX[1:] # Give each 5 minute tick the technical indicator vals from last tick
XX_difference = XX[:-1] - XX_generated 
XX=XX[:-1]
yy = yy[:-1] # Take off the oldest prediction value to match

if generate_features == True:
    XX = np.hstack((XX,XX_generated,XX_difference)) # Tack them on to catalogue and take off the oldest tick to compensate the shift

if makedata_convtest == True:
    XX_conv=[]
    for i in range(len(XX)-nb_ticks_history):
        XX_conv.append(XX[i:i+nb_ticks_history])
    XX=np.array(XX_conv)
    yy=yy[0:-nb_ticks_history]

if shuffle_whole_cat == True:
    XX,yy=shuffle(XX,yy)

XX = XX[0:n_cat] # Cut catalogue down to n_cat
yy = yy[0:n_cat]

if price_dir_only == True:
    yy = np.array(yy > 0).astype('int')
    yy = keras.utils.to_categorical(yy, 2)

XX_train,XX_test,yy_train,yy_test = \
train_test_split(XX,yy,test_size=test_size,shuffle=shuffle_cats) # Split data into train and test set with test_size as ratio

# SCALING
tr_means,tr_stds=[],[]
for i in range(XX_train.shape[-1]):
    if makedata_convtest ==True:
        tr_means.append(np.mean(XX_train[:,0,i]))
        tr_stds.append(np.std(XX_train[:,0,i]))
    else:
        tr_means.append(np.mean(XX_train[:,i]))
        tr_stds.append(np.std(XX_train[:,i]))

XX_train = zcmn_scaling(XX_train,tr_means,tr_stds)
XX_test = zcmn_scaling(XX_test,tr_means,tr_stds)

#if makedata_convtest == True:
#    XX_train=XX_train.reshape(XX_train.shape[0],XX_train.shape[1],XX_train.shape[2],1)
#    XX_test=XX_test.reshape(XX_test.shape[0],XX_test.shape[1],XX_test.shape[2],1)

#yy_train = np.round(yy_train,decimals=8)

# NN MODEL SETUP
if makedata_convtest == True:
    model = Sequential()
    model.add(Conv1D(input_shape=(nb_ticks_history, XX_train.shape[-1]),filters=8,kernel_size=3, strides=4))
    #model.add(Dense(input_dim = (XX_train.shape[1]), output_dim = 1024))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.ELU(alpha=1.))
    model.add(MaxPooling1D(pool_size=(2)))
#    model.add(Conv1D(filters=32,kernel_size=3, strides=4))
#    ##model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
#    model.add(keras.layers.advanced_activations.ELU(alpha=1.))
#    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Flatten())
    model.add(Dense(4))
    #model.add(keras.layers.advanced_activations.ELU(alpha=1.))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    model.add(Dropout(0.50))
#    #model.add(Dense(input_dim = 1024, output_dim = 256))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    #model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
#    #model.add(Dropout(0.5))
#    model.add(Dense(256))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.ELU(alpha=1.))
#    #model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None))
#    model.add(Dropout(0.5))

else:
    model = Sequential()
    model.add(Dense(input_dim = XX_train.shape[1], output_dim = 128))
    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    model.add(keras.layers.advanced_activations.ELU(alpha=1.))
    #model.add(Dropout(0.5))
    model.add(Dense(input_dim = 128, output_dim = 2))
    model.add(keras.layers.advanced_activations.PReLU(init='zeros', weights=None))
    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
#    model.add(Dense(input_dim = 256, output_dim = 1024))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.ELU(alpha=1.))
#    model.add(Dropout(0.5))
#    model.add(Dense(input_dim = 256, output_dim = 256))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
#    model.add(Dropout(0.5))
#    model.add(Dense(input_dim = 256, output_dim = 256))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.ELU(alpha=1.))
#    model.add(Dropout(0.5))
#    model.add(Dense(input_dim = 256, output_dim = 1024))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
#    model.add(Dropout(0.5))
#    model.add(Dense(input_dim = 256, output_dim = 256))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.ELU(alpha=1.))
#    model.add(Dropout(0.5))
#    model.add(Dense(input_dim = 256, output_dim = 256))
#    #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
#    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
#    model.add(Dropout(0.5))
if price_dir_only == True:
    model.add(Dense(2,activation='softmax'))
else:
    model.add(Dense(input_dim = 2, output_dim = 1))

opt = keras.optimizers.Adam(lr=.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#opt=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
#opt = keras.optimizers.SGD(lr=0.0005,momentum=0.8,decay=0.01)
if price_dir_only == True:
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
else:
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

if load_old_model == True:
    model=load_model(modelname+".hdf5")

if run_training == True:
    model.summary()
    model.fit(XX_train, yy_train,batch_size=batch_size,epochs=epochs\
    ,validation_data=(XX_test, yy_test),shuffle=shuffle_cats)

if run_pred == True:
    results=model.predict(XX_test)
    results=results[:,1]

if price_dir_only == True:
    yy_test=yy_test[:,1]
    logger.info('Percent of price direction correct: %0.4f'%(np.float(sum(sum([results>.5] == yy_test))/np.float(len(yy_test)))))
else:
    mse=metrics.mean_squared_error(yy_test,results) # Get MSE		
    logger.info('Training data absolute mean of percentage price change (what we''re predicting: %0.4f'%np.mean(np.abs(yy_train)))
    logger.info("Model MSE (of predict cat): %s"%mse)		
    logger.info("Model RMSE (of predict cat): %s"%np.sqrt(mse))
    percent_diff = results - yy_test
    plt.plot(range(len(percent_diff)), results/yy_test) # Silly plot I think is useful but it's not. I look at it because I'm lazy.
    plt.ylim(-10,10)
    pos_res=results>0
    pos_test=yy_test>0
    n_test=test_size*n_cat
    n_direction_corr = sum(pos_res==pos_test)
    logger.info('Percent of price direction correct: %0.4f'%(np.float(n_direction_corr)/np.float(n_test)))

# PUT THIS IN IT'S OWN FUNCTION THAT CHECKS FOR TICK UPDATES AND COMPUTES THIS
# GENERATE FEATURES FOR LATEST 5 MIN TICK AND PREDICT THE FUTURE
latest_tick_all = df[-1:].values # get all features from latest 5 min bin or tick, the one we're going to predict the price change of
tick_before_latest_all = df[-2:-1].values # get all features from 5 min bin or tick before that
latest_tick = latest_tick_all.T[onlyusemask] # filter out features we want
tick_before_latest= tick_before_latest_all.T[onlyusemask]
latest_tick = latest_tick.T # Stupid transpose
tick_before_latest = tick_before_latest.T
tick_difference = latest_tick - tick_before_latest # generate more features. the difference in features between ticks
if generate_features ==True:
    XX_latest = np.hstack((latest_tick,tick_before_latest,tick_difference)) # Stack all the features together
else: XX_latest=latest_tick
XX_latest = zcmn_scaling(XX_latest,tr_means,tr_stds) # Scale it according to the training set's stats
fut_prediction = model.predict(XX_latest) # PREDICT THE FUTURE

# OLD ARBITRAGE LOGIC - probably not needed for this code.
#testwallet=1
#def forward_search():
#    global n_forw,n_forw_bad, for_run,result_for
#    n_forw=0
#    n_forw_bad=0
#    for_run=1
#    while for_run==1:    
#        time.sleep(0.3)
#        result_for=(((((market_orders[0][1]*testwallet)*.998)/market_orders[1][3])*.998)*market_orders[2][1])*.998
#        if result_for > 1:
#            n_forw=n_forw+1
#            print 'FORWARDS (Bitfinex)'
#            print time.ctime()
#        else:
#            n_forw_bad=n_forw_bad+1
#        
#
##BACKWARDS
#def backward_search():
#    global n_back,n_back_bad, bac_run,result_bac
#    n_back=0
#    n_back_bad=0
#    bac_run=1
#    while bac_run==1:
#        time.sleep(0.3)
#        result_bac=(((((testwallet/market_orders[2][3])*.998)*market_orders[1][1])*.998)/market_orders[0][3])*.998
#        if result_bac > 1:
#            n_back=n_back+1
#            print 'BACKWARDS (Bitfinex)'
#            print time.ctime()
#        else:
#            n_back_bad=n_back_bad+1
#
#time.sleep(5)
#fsT=threading.Thread(target=forward_search)
#bsT=threading.Thread(target=backward_search)
#fsT.daemon=True
#bsT.daemon=True
#fsT.start()
#bsT.start()