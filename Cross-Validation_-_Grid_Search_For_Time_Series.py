"""
Grid Search for Time series does not for right now.
Because GridSearchCV require that train data set and test data set have same size
but the TimeSeriesSplit function will split original data set into test data and
train data set with different size

Another reason is the the GridSearchCV cannot clone the base model we set up as
the base model fro Grid Search
"""


"""
# DataFrame.iloc is selecting data based on position,
# while DataFrame.loc is selecing data based on index which means input should
# be the same as index, then we can extract data
"""

from __future__ import division #handle the 1/12 = 0 not 0.83 problem
import os
import sys

import pandas as pd
import pandas_datareader.data as web
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt
import matplotlib as mpl

import datetime #deal with the start and end time

Start = datetime.datetime(2012, 1, 1)
End = datetime.datetime(2017, 6, 26)
# This model only works when there is one stock and one item like close or open
Stocks = ['MSFT']
# ['Close'] is for just extracting Close stock price
# [['Volume','Close']] is for extract close stock price and Volume
Data = web.DataReader(Stocks, 'google', Start, End)['Close']
Data = Data.resample('1M').mean()

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

Model = smt.ARIMA(Data['MSFT'], order=(1,0,0))
param_grid = [{'order':[(1,0,0)},{'order':[(3,1,2)]}]

tscv = TimeSeriesSplit(n_splits=3)

for Data_Train, Data_Test in tscv.split(Data):
    DataTr = Data.iloc[Data_Train]
    DataTr = DataTr[:16]
    DataTe = Data.iloc[Data_Test]
    CV_Model = GridSearchCV(estimator=Model, param_grid=param_grid, cv= 5).fit(DataTr['MSFT'], DataTe['MSFT'])
    #print CV_Model.summary()
