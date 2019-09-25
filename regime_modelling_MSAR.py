import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


'''
Get the data and calculate returns
'''
nifty = pd.read_csv('nifty.csv', index_col=0, parse_dates=True)['Nifty']# Get nifty prices
nifty_ret = nifty.resample('W').last().pct_change().dropna()# Get weekly returns
nifty_ret.plot(title='Weekly returns', figsize=(12, 3)) # Plot the dataset


'''
Test for stationarity
'''
adfuller(nifty_ret.dropna())


'''
Fit the model
'''
mod_kns = sm.tsa.MarkovRegression(nifty_ret.dropna(), k_regimes=3, trend='nc', switching_variance=True)
res_kns = mod_kns.fit()
res_kns.summary()


'''
Plot smoothed probabilities
'''
fig, axes = plt.subplots(3, figsize=(10,7))ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of a low-variance regime for stock returns')ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a medium-variance regime for stock returns')ax = axes[2]
ax.plot(res_kns.smoothed_marginal_probabilities[2])
ax.set(title='Smoothed probability of a high-variance regime for stock returns')fig.tight_layout()
