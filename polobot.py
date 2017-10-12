# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:30:36 2017

@author: xangma
"""

from __future__ import print_function

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

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split

#SETTINGS
MLA = 'sklearn.ensemble.RandomForestRegressor'                             # Which MLA to load
MLAset = {'n_estimators': 256, 'n_jobs': 4, 'verbose': 1}
# features to choose from:
# array(['close', 'high', 'low', 'open', 'quoteVolume', 'volume', 'weightedAverage', 'sma', 'bbtop', 'bbbottom', 'bbrange', 'bbpercent', 'emaslow', 'emafast', 'macd', 'rsi', 'bodysize', 'shadowsize', 'percentChange']
onlyuse = ['weightedAverage', 'sma', 'bbtop', 'bbbottom', 'bbrange','bbpercent', 'emaslow', 'emafast', 'macd', 'rsi']
test_size = 0.1
shuffle_cats = False



logger = logging.getLogger(__name__)

def rsi(df, window, targetcol='weightedAverage', colname='rsi'):
    """ Calculates the Relative Strength Index (RSI) from a pandas dataframe
    http://stackoverflow.com/a/32346692/3389859
    """
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
        df = rsi(df, window // 5)
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
# websocket.enableTrace(True)
ticker = dictTicker()
try:
    ticker.start()
#    for i in range(3):
#        sleep(5)
#        pprint.pprint(ticker('USDT_BTC'))
except Exception as e:
    logger.exception(e)
#ticker.stop()

logging.getLogger("poloniex").setLevel(logging.INFO)
logging.getLogger('requests').setLevel(logging.ERROR)
api = Poloniex(jsonNums=float)

def updateChart(market):
    df = Chart(api, market).dataFrame()
    df.dropna(inplace=True)
    return df

def get_function(function_string):
    import importlib
    module, function = function_string.rsplit('.', 1)
    module = importlib.import_module(module)
    function = getattr(module, function)
    return function

df = updateChart('BTC_ETH')

ts = (df.index[-1] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
latest_time=datetime.utcfromtimestamp(ts)

n_train=10000
features = df.columns.values.astype('str') # get column names from dataframe
XX = df.values # Get column values from dataframe
XX = np.flipud(XX) # makes latest 5 minute tick XX[0] instead of XX[-1] (flips the array so most recent is at top)
percentchange = XX[:,18] # Takes percentage train as variable to predict
XX = XX[1:,:-1] # Takes off latest tick, as we're going to shift the 'percent change since last tick' down. It will become the 'percent change for the next tick'. Also removes the percentage change from the features list 

yy=percentchange[:-1]
# NEED TO CUT DOWN FEATURE COLUMNS HERE
onlyusemask= [x in onlyuse for x in features]
XX = XX.T[onlyusemask]
XX = XX.T
features = features[onlyusemask]
# FEATURE GENERATION. Take feature values from last tick
XX_generated = XX[1:]
XX = np.hstack((XX[:-1],XX_generated))
yy = yy[:-1]

XX_train,XX_test,yy_train,yy_test = train_test_split(XX,yy,test_size=test_size,shuffle=shuffle_cats)

MLA = get_function(MLA) # Pulls in machine learning algorithm from settings
clf = MLA().set_params(**MLAset)

logger.info("Training model ... ")
clf.fit(XX_train,yy_train)
results = clf.predict(XX_test)
mse=metrics.mean_squared_error(yy_test,results)

logger.info("Model MSE: %s"%mse)
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