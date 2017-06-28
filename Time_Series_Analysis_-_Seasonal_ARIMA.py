import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

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



df = pd.read_csv('C:\Users\wye\Desktop\Work\Python2\DataSet\portland-oregon-average-monthly-.csv',
                    index_col=0)
df.index.name=None
df.reset_index(inplace=True)
df.drop(df.index[114], inplace=True)

start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,114)]
df['index'] =date_list
df.set_index(['index'], inplace=True)
df.index.name=None

df.columns= ['riders']
df['riders'] = df.riders.apply(lambda x: int(x)*100)

# df.riders.plot(figsize=(12,8), title= 'Monthly Ridership', fontsize=14)
# plt.savefig('month_ridership.png', bbox_inches='tight')

# decomposition = seasonal_decompose(df.riders, freq=12)
# #fig = plt.figure()
# fig = decomposition.plot()
# fig.set_size_inches(12, 5)
#
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
# plt.show()


def test_stationarity(timeseries):
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)
    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

# Null Hypothesis is that the time series is not stationary
# Alternative Hypothesis is that the time seires is stationary
# if the absolute value of Test Statistic is larger than critical value
# then we reject the null Hypothesis, and the time series is stationary
# if the absolute value of  Test Statistic is smaller than critical value
# then we cannot reject the null Hypothesis, and the time series is non stationary

# test_stationarity(df.riders)

"""
This code can find out the model with lowest aic, but it will take long time

# best_aic = np.inf
# best_summary = None
# best_mdl = None
#
# pq_rng = range(5) # [0,1,2,3,4]
# d_rng = range(2) # [0,1]
# for i in pq_rng:
#     for d in d_rng:
#         for j in pq_rng:
#             for si in pq_rng:
#                 for sd in d_rng:
#                     for sj in pq_rng:
#                         try:
#                             tmp_mdl = sm.tsa.statespace.SARIMAX(df.riders,
#                                     trend='n', order=(i,d,j),
#                                     seasonal_order=(si,sd,sj,12)).fit(maxiter = 70, disp = False)
#                             tmp_aic = tmp_mdl.aic
#                             print tmp_aic, i,d,j, si, sd, sj
#                             tmp_summary = tmp_mdl.summary()
#                             if tmp_aic < best_aic:
#                                 best_aic = tmp_aic
#                                 best = tmp_mdl
#                                 best_mdl = tmp_mdl.summary()
#                         except: continue
#
#
#
#
# print (best_mdl)
"""

S_model= sm.tsa.statespace.SARIMAX(df.riders, trend='n',
        order=(0,1,0), seasonal_order=(3,1,0,12))
S_Result = S_model.fit(disp = False)
print results.summary()

df['forecast'] = S_Result.predict(start = 102, end= 114, dynamic= True)
df[['riders', 'forecast']].plot(figsize=(12, 8))
plt.show()

# show the difference between forecast and Observations
npredict =df.riders['1982'].shape[0]
fig, ax = plt.subplots(figsize=(12,6))
npre = 12
ax.set(title='Ridership', xlabel='Date', ylabel='Riders')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'riders'], 'o', label='Observed')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'forecast'], 'g', label='Dynamic forecast')
legend = ax.legend(loc='lower right')
legend.get_frame().set_facecolor('w')
plt.show()

# out of sample prediction
start = datetime.datetime.strptime("1982-07-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= df.columns)
df = pd.concat([df, future])
df['forecast'] = results.predict(start = 114, end = 125, dynamic= True)
df[['riders', 'forecast']].ix[-24:].plot(figsize=(12, 8))
plt.show()
