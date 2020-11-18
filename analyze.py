import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.api import qqplot
import seaborn as sns

def np_autocorr(x, t):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

def autocorr():
    df = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    df = df.reindex(index=df.index[::-1])
    df.index.freq = 'H' # Hourly data.

    df0 = df.loc['2016-01-1' : '2016-2-1']
    ts0 = np.asarray(df[['Electricity Demand in the State of California']].values)
    ts0 = ts0.flatten()
    ts0, l = stats.boxcox(ts0)

    df1 = df.loc['2016-01-1' : '2019-1-01-00']
    ts1 = np.asarray(df[['Electricity Demand in the State of California']].values)
    ts1 = ts1.flatten()
    ts1, l = stats.boxcox(ts1)


    #s = np.array([])
    #s = s.flatten()
    #for i in range(1, len(ts) - 1):
    #    if i < 24 * 31 and i % 24 == 0:
    #        val = np_autocorr(ts, i)
    #        s = np.append(s, val)
    #x = np.linspace(0, 100 * len(s), len(s))
    #sns.set(rc={'figure.figsize':(16, 4)})
    #sns.scatterplot(x, s)
    #plt.show()

    #series = df.T.squeeze()
    #s = pd.Series()
    #s_index = 0
    #for i in range(0, len(series)):
    #    if i % 24 == 0: 
    #        s = s.append(pd.Series(series.autocorr(lag=i), index=[s_index]))
    #        s_index = s_index + 1
                

    #sns.set(rc={'figure.figsize':(14, 4)})
    #df.plot(linewidth=1)
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharey=True)
    fig.suptitle('')
    sns.set(rc={'figure.figsize':(12, 4)})
    plot_acf(ts0, lags=24*31, ax=axes[0])
    sns.set(rc={'figure.figsize':(18, 5)})
    plot_acf(ts1, lags=365 * 24, linewidth=0.01, ax=axes[1])
    
    #plt.xlabel("Time [Hour]")
    plt.xlabel("Lag [Hour]")
    #plt.xlabel("Lag [Day]")
    #plt.ylabel("Demand [Megawatt]")
    plt.ylabel("Auto-correlation coefficient")
    plt.show()
     

def decompose():
    df = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    df = df.reindex(index=df.index[::-1])
    df.index.freq = 'H' # Hourly data.
    df = df.loc['2015-01-1' : '2019-12-31']

    decomposed = STL(df, seasonal=25, period=501).fit()
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    rest = decomposed.resid
    
    print(str(rest.var() / (rest.var() + trend.var())))
    print(str(rest.var() / (rest.var() + seasonal.var())))
    trend.plot()

 
    #sns.set(rc={'figure.figsize':(30, 3)})
    #components = seasonal_decompose(df['Electricity Demand in the State of California'], model='additive', period=24)
    #trend = components.trend
    #seasonal = components.seasonal
    #rest = components.resid

    #print(str(1 - rest.var() / (rest.var() + trend.var())))
    #print(str(1 - rest.var() / (rest.var() + seasonal.var())))



    #components.plot()
    #components = seasonal_decompose(df['Electricity Demand in the State of California'], model='multiplicative', period=24).seasonal
    #components.plot()
    #components = seasonal_decompose(df['Electricity Demand in the State of California'], model='multiplicative', period=24 * 7)
    #components.plot()
    #components = seasonal_decompose(df['Electricity Demand in the State of California'], model='multiplicative', period=24 * 7 * 52)
    #components.plot()
    plt.show()

def plot_stats():
    df = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    df = df.reindex(index=df.index[::-1])
    df.index.freq = 'H' # Hourly data.
    df = df.loc['2016-01-1' : '2016-01-31']

    #sns.set(rc={'figure.figsize':(5, 2)})
    #df.plot()
 
    sns.histplot(df)
    df.plot.hist(by=None, bins=15)
    plt.show()

def main():
    #plot_stats()
    #decompose()
    autocorr()

if __name__ == '__main__':
    main()
