import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as st
from math import exp
import random
from datetime import datetime
import datetime as dt
import yfinance as yf

#This code is part of the Market Implied PD Extraction explanatory Report published by AUEBs' Investment and Finance Club.

ticker='PPC.AT'
#Starting data preparation process - Downloading Historical Equity Market Prices for PPC.AT
def data(ticker,start,end):
    data = pd.DataFrame(yf.download(ticker,start,end)['Close'])
    data['Close'].loc[:'2021-11-16'] = data['Close'].loc[:'2021-11-16'] * 232e06
    data['Close'].loc['2021-11-17':] = data['Close'].loc['2021-11-17':] * 382e06
    data.rename(columns={'Close': 'Equity MV'}, inplace=True)
    return data
end = datetime(2024, 1, 24) - dt.timedelta(days=23)
start = end - dt.timedelta(days=2547)
data_matrix = data(ticker,start,end)

#Input of PPC.AT Liabilities data and organizing them into a singe DataFrame, alongside Equity Prices
total_liabilities = list()
default_point = list()
start_dp = 6.234e09
for i in range(248):
   total_liabilities.append(9.740e9)
   start_dp = start_dp*(1 +0.00020)
   default_point.append(start_dp)
for i in range(248):
    total_liabilities.append(10.0098e9)
    start_dp = start_dp*(1 +0.00014)
    default_point.append(start_dp)
for i in range(248):
    total_liabilities.append(10.531e9)
    start_dp = start_dp*(1+8.1656e-05)
    default_point.append(start_dp)
for i in range(248):
    total_liabilities.append(10.599e9)
    start_dp = start_dp * (1+4.9843e-05)
    default_point.append(start_dp)
for i in range(248):
    total_liabilities.append(12.699e9)
    start_dp = start_dp * (1 + 0.00084)
    default_point.append(start_dp)
for i in range(248):
    total_liabilities.append(14.846e9)
    start_dp = start_dp * (1+0.00063)
    default_point.append(start_dp)
for i in range(248):
    total_liabilities.append(14.500e9)
    default_point.append(start_dp)

#Importing and organizing Risk Free Data (source: marketwatch.com)
rfr = pd.read_excel('ggb yields.xlsx')
rfr['Date'] = pd.to_datetime(rfr['Date'])
rfr= rfr.sort_index(ascending=False)
rfr.set_index('Date', inplace=True)
rfr['Close'] = np.log(1+rfr['Close'])
data_matrix['Risk Free Log Rate'] =rfr['Close'].iloc[0:1736].values
data_matrix['Liabilities BV'] = total_liabilities
data_matrix['Default Point'] = default_point

#Estimating initial Asset Value and Asset Sigma via iteration process (minimizing the SSE of guessed Asset Value and theoretical Asset Value implied by BLS formula. In the first iteration we guess Asset Value as the BV of Assets).
def AssetValue_Sigma_Estimation(data_matrix):
    initial_asset_guess = (data_matrix['Equity MV'] + data_matrix['Liabilities BV']).to_numpy()
    equity = data_matrix['Equity MV'].to_numpy()
    dp = data_matrix['Default Point'].to_numpy()
    rf = data_matrix['Risk Free Log Rate'].to_numpy()
    T = 1
    iterations = 6
    sse = 100
    asset_series = list()
    assets = initial_asset_guess
    for i in range(iterations):
        asset_series.append(assets)
        if sse <= 1e-10:
            break
        else:
            assets_std = np.std([np.log(assets[i + 1] / assets[i]) for i in range(len(assets) - 1)]) * np.sqrt(252)
            d2 = np.log((assets/dp)+(rf-(assets_std ** 2)/2)*T)/assets_std*np.sqrt(T)
            d1 = np.log((assets/dp)+(rf+(assets_std ** 2)/2)*T)/assets_std*np.sqrt(T)
            Asset_Value_estimate = list()
            for i in range(len(assets)):
                Asset_Value_estimate.append((equity[i]+dp[i]*exp(-rf[i]*T)*st.norm.cdf(d2[i]))/st.norm.cdf(d1[i]))
            sse = np.sum([(i - j)**2 for i, j in zip(assets,Asset_Value_estimate)])
            assets = Asset_Value_estimate
    iteration_results = (pd.DataFrame(asset_series)).T
    optimized_asset_ts = iteration_results.iloc[:, -1]
    return optimized_asset_ts

last_year_dm = data_matrix.loc['2022-12-29':'2023-12-29']
optimized_asset_ts = AssetValue_Sigma_Estimation(last_year_dm)
print(last_year_dm)
print(optimized_asset_ts)
asse_std = np.std([np.log(optimized_asset_ts[i + 1] / optimized_asset_ts[i]) for i in range(len(optimized_asset_ts) - 1)]) * np.sqrt(252)
equity_std = np.std([np.log(last_year_dm['Equity MV'].iloc[i+1]/last_year_dm['Equity MV'].iloc[i]) for i in range(len(optimized_asset_ts) - 1)])* np.sqrt(252)
print(equity_std)
print(asse_std)

def PD1Y(asset_initial, sigma, default_point, rf, T):
   d2 = (np.log(asset_initial/default_point)+(rf - (sigma**2)/2)*T)/sigma*np.sqrt(T)
   PD = st.norm.cdf(-d2)
   return PD

Ao = optimized_asset_ts[len(optimized_asset_ts) - 1]
astd = np.std([np.log(optimized_asset_ts[i+1]/optimized_asset_ts[i]) for i in range(len(optimized_asset_ts)-1)])*np.sqrt(252)
def_point = last_year_dm['Default Point'].to_numpy()[-1]
rskfree = (last_year_dm['Risk Free Log Rate'].to_numpy()[-1])
T = 1
result = round(PD1Y(Ao, astd, def_point, rskfree, T)*100,10)
print(f'Probability of Default over the next year for {ticker} is: {result} %')








