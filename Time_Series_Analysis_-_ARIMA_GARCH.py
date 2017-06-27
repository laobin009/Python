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
# Data = Data_M.pct_change().dropna()
# log_re = np.log(Data_M/Data_M.shift(1)).dropna()

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
    return



best_aic = np.inf
best_order = None
best_mdl = None

pq_rng = range(5) # [0,1,2,3,4]
d_rng = range(2) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl = smt.ARIMA(Data['MSFT'],
                        order=(i,d,j)).fit(method='mle', trend='nc',
                        disp = False, maxiter = 30)
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue

# see the plot of residual of the best model
tsplot(best_mdl.resid, lags=30)

# Forecast
Forecast, Stderr, conf_int = best_mdl.forecast(steps = 12, alpha = 0.05)
idx = pd.date_range(Data.index[-1], periods=12, freq='M')
FT = pd.DataFrame(np.column_stack([Forecast, Stderr, conf_int ]),
                    index=idx, columns=['Forecast', 'Standard_Error',
                    'Lower_CI_95', 'Upper_CI_95'])


print (best_mdl.summary())
print ('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

print FT.head(n = 20)
# print best_mdl.resid
plt.show()

# # observe the residual to determine if we meet the constant variance assumption
# index = pd.date_range(Data.index[0], Data.index[-1])
# plt.scatter(index[1:],best_mdl.resid)


# # Garch model
# best_G_aic = np.inf
# best_G_order = None
# best_G_mdl = None
# for i in pq_rng:
#     for d in d_rng:
#         for j in pq_rng:
#             try:
#                 T_Arch = arch_model(Data, p=i, o=d, q=j, dist='StudentsT').fit(update_freq=5, disp='off')
#                 T_G_aic = T_Arch.aic
#                 if T_G_aic < best_G_aic:
#                     best_G_aic = T_G_aic
#                     best_G_order = (i, d, j)
#                     best_G_mdl = T_Arch
#             except: continue
#
#
#
# print (best_G_mdl.summary())
# print ('aic: {:6.5f} | order: {}'.format(best_G_aic, best_G_order))
