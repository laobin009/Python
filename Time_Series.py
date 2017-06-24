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

from __future__ import division #handle the 1/12 = 0 not 0.83 problem

Start = datetime.datetime(2007, 1, 1)
End = datetime.datetime(2015, 1, 1)
# This model only works when there is one stock and one item like close or open
Stocks = ['SPY']
# ['Close'] is for just extracting Close stock price
# [['Volume','Close']] is for extract close stock price and Volume
Data = web.DataReader(Stocks, 'google', Start, End)['Close']
lrets = np.log(Data/Data.shift(1)).dropna()

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

#####################################################################
# plot of discrete white noise
np.random.seed(1)
randser = np.random.normal(size=1000)
tsplot(randser, lags=30)
plt.show()
#####################################################################

#####################################################################
# Random Walk without a drift
np.random.seed(1)
n_samples = 1000
x = w = np.random.normal(size=n_samples)
for t in range(n_samples):
    x[t] = x[t-1] + w[t]

tsplot(np.diff(Data['SPY']), lags=30)
plt.show()
#####################################################################

#####################################################################
# Simulate an AR(1) process with alpha = 0.6
np.random.seed(1)
n_samples = int(1000)
a = 0.6
x = w = np.random.normal(size=n_samples)
for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]

tsplot(x, lags=30)
plt.show()
#####################################################################

#####################################################################
# Fit an AR(p) model to simulated AR(1) model with alpha = 0.6
mdl = smt.AR(x).fit(maxlag=30, ic='aic', trend='nc')
est_order = smt.AR(x).select_order(
    maxlag=30, ic='aic', trend='nc')

true_order = 1
print ('\nalpha estimate: {:3.5f} | best lag order = {}'
  .format(mdl.params[0], est_order))
print ('\ntrue alpha = {} | true order = {}'
  .format(a, true_order))
#####################################################################

#####################################################################
# Simulate an AR(2) process
n = int(1000)
alphas = np.array([.666, -.333])
betas = np.array([0.])

# Python requires us to specify the zero-lag value which is 1
# Also note that the alphas for the AR model must be negated
# We also set the betas for the MA equal to 0 for an AR(p) model
# For more information see the examples at statsmodels.org
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
tsplot(ar2, lags=30)

# Fit an AR(p) model to simulated AR(2) process

max_lag = 10
mdl = smt.AR(ar2).fit(maxlag=max_lag, ic='aic', trend='nc')
est_order = smt.AR(ar2).select_order(
    maxlag=max_lag, ic='aic', trend='nc')

true_order = 2
print ('\ncoef estimate: {:3.4f} {:3.4f} | best lag order = {}'
  .format(mdl.params[0],mdl.params[1], est_order))
print ('\ntrue coefs = {} | true order = {}'
  .format([.666,-.333], true_order))


# Select best lag order for MSFT returns

max_lag = 30
mdl = smt.AR(lrets['SPY']).fit(maxlag=max_lag, ic='aic', trend='nc')
est_order = smt.AR(lrets['SPY']).select_order(
    maxlag=max_lag, ic='aic', trend='nc')

print ('best estimated lag order = {}'.format(est_order))
#####################################################################

#####################################################################
# Simulate an MA(1) process

n = int(1000)

# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.6])

# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)

# Fit the MA(1) model to our simulated time series
# Specify ARMA model with order (p, q)

max_lag = 30
mdl = smt.ARMA(ma1, order=(0, 1)).fit(
    maxlag=max_lag, method='mle', trend='nc', disp = False)

print (mdl.summary())

#####################################################################

#####################################################################
# Here we sue mdl.resid, we want to know whether the residual is white noise
# if ACF and PACF show no correlation, then it is white noise
# which is good.
max_lag = 30
Y = lrets['SPY']
mdl = smt.ARMA(Y, order=(0, 3)).fit(
    maxlag=max_lag, method='mle', trend='nc',disp = False)
print (mdl.summary())
tsplot(mdl.resid, lags=max_lag)
plt.show()


#####################################################################

#####################################################################
# Simulate an ARMA(3, 2) model with alphas=[0.5,-0.25,0.4] and betas=[0.5,-0.3]
max_lag = 30
n = int(5000)
burn = 2000 # number of samples to discard before fit

alphas = np.array([0.5, -0.25, 0.4])
betas = np.array([0.5, -0.3])

ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma32 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
tsplot(arma32, lags=max_lag)

# pick best order by aic
# smallest aic value wins
best_aic = np.inf # np.inf return positive infinity
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(arma32, order=(i, j)).fit(method='mle', trend='nc'
                                ,disp = False)
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print ('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

#####################################################################

#####################################################################
# Fit ARMA model to SPY returns

best_aic = np.inf
best_order = None
best_mdl = None

rng = range(5) # [0,1,2,3,4,5]
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(lrets['SPY'], order=(i, j)).fit(
                method='mle', trend='nc', disp = False
            )
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue

print (best_mdl.summary())
print ('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

#####################################################################

#####################################################################
# Fit ARIMA(p, d, q) model to SPY Returns
# pick best order and final model based on aic
best_aic = np.inf
best_order = None
best_mdl = None

pq_rng = range(5) # [0,1,2,3,4]
d_rng = range(2) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl = smt.ARIMA(lrets['SPY'],
                        order=(i,d,j)).fit(method='mle', trend='nc',
                        disp = False)
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue

tsplot(best_mdl.resid, lags=30)
print (best_mdl.summary())
print ('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
plt.show()
#####################################################################

#####################################################################
# Create a 21 day forecast of SPY returns with 95%, 99% CI
n_steps = 21

Forecast, err95, ci95 = best_mdl.forecast(steps=n_steps) # 95% CI
_, err99, ci99 = best_mdl.forecast(steps=n_steps, alpha=0.01) # 99% CI

idx = pd.date_range(Data.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([Forecast, ci95]),
                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]),
                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
print (fc_all.head())

#####################################################################

#####################################################################

# Plot 21 day forecast for SPY returns

plt.style.use('bmh')
fig = plt.figure(figsize=(9,7))
ax = plt.gca()

ts = lrets['SPY'].iloc[-500:].copy()
ts.plot(ax=ax, label='Spy Returns')
# in sample prediction
# Difference between predict and forecast is in predict we use sample and
# observe how does our model perform and see the difference betwwen sample and
# prediction(model output), in forecast, we cannot compare.

# here  we use ts sample from the first element to the last element.
pred = best_mdl.predict(ts.index[0], ts.index[-1])
pred.plot(ax=ax, style='r-', label='In-sample prediction')

styles = ['b-', '0.2', '0.75', '0.2', '0.75']
fc_all.plot(ax=ax, style=styles)
plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95,
                    color='gray', alpha=0.7)
plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99,
                    color='gray', alpha=0.2)
plt.title('{} Day SPY Return Forecast\nARIMA{}'.format(n_steps, best_order))
plt.legend(loc='best', fontsize=10)

#####################################################################

#####################################################################
# Simulate ARCH(1) series
# Var(yt) = a_0 + a_1*y{t-1}**2
# if a_1 is between 0 and 1 then yt is white noise

np.random.seed(13)

a0 = 2
a1 = .5

y = w = np.random.normal(size=1000)
Y = np.empty_like(y)

for t in range(len(y)):
    Y[t] = w[t] * np.sqrt((a0 + a1*y[t-1]**2))

# simulated ARCH(1) series, looks like white noise
tsplot(Y, lags=30)
#####################################################################

#####################################################################
# SIMULATED ARCH(1)**2 PROCESS

np.random.seed(13)

a0 = 2
a1 = .5

y = w = np.random.normal(size=1000)
Y = np.empty_like(y)

# Compare to the ARCH(1) PROCESS, we remove the w[t]
for t in range(len(y)):
    Y[t] =  np.sqrt((a0 + a1*y[t-1]**2))

# simulated ARCH(1) series, looks like white noise
tsplot(Y, lags=30)

#####################################################################

#####################################################################
# Simulating a GARCH(1, 1) process

np.random.seed(2)

a0 = 0.2
a1 = 0.5
b1 = 0.3

n = 10000
w = np.random.normal(size=n)
eps = np.zeros_like(w)
sigsq = np.zeros_like(w)

for i in range(1, n):
    sigsq[i] = a0 + a1*(eps[i-1]**2) + b1*sigsq[i-1] # here is AR + MA(q)
    eps[i] = w[i] * np.sqrt(sigsq[i]) # here is AR(p)

tsplot(eps, lags=30)
tsplot(sigsq, lags=30)
plt.show()
#####################################################################

#####################################################################

# Fit a GARCH(1, 1) model to our simulated EPS series
# We use the arch_model function from the ARCH package

am = arch_model(sigsq)
res = am.fit(update_freq=5)
print (res.summary())

#####################################################################

#####################################################################
TS = lrets['SPY'].ix['2010':'2015']
best_aic = np.inf
best_order = None
best_mdl = None
pq_rng = range(5) # [0,1,2,3,4]
d_rng = range(2) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                    method='mle', trend='nc', disp = False
                )
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue




print ('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

# Notice I've selected a specific time period to run this analysis

tsplot(best_mdl.resid, lags = 30)
plt.show()


#####################################################################

#####################################################################

# Using student T distribution usually provides better fit
am = arch_model(TS, p=0, o=1, q=1, dist='StudentsT')
res = am.fit(update_freq=5, disp='off')
print (res.summary())
