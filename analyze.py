import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

def np_autocorr(x, t):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

def autocorr(data):
    series = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, index_col=0)
    series = series.reindex(index=series.index[::-1])
    series.index.freq = 'H' # Hourly data.
    for i in series.index:
        print(series.Date[i])
    return
    p = plot_acf(series, lags=24 * 365)
    plt.xlabel("Lag value")
    plt.ylabel("Autocorrelation")

    plt.show()
    #print("Not box cox:")
    print(data['Values'].autocorr(lag=24))
    print(data['Values'].autocorr(lag=24 * 7))
    print(data['Values'].autocorr(lag=24 * 365))

    #ts = np.asarray(data[['Values']].values)
    #ts = ts.flatten()
    #ts = stats.boxcox(ts)[0]
    #print("Box cox:")
    #print(np_autocorr(ts, 24))

def decompose(data):
    df = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, index_col=0, parse_dates=['Category'])[0 : 24*31 * 3]
    components = seasonal_decompose(df['Values'], model='multiplicative', period=24)
    components.plot()
    components = seasonal_decompose(df['Values'], model='additive', period=24 * 7)
    components.plot()
    plt.show()

def print_stats(data):
    stats=pd.DataFrame()
    stats["mean"]=data.mean()
    stats["Std.Dev"]=data.std()
    stats["Var"]=data.var()
    print(stats)

def plot_stats(data):
    data.plot.hist(by=None, bins=1000)
    plt.show()

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', infer_datetime_format=True)
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.

    data_split = [data.loc['2015-07-1' : '2015-10-31'].to_numpy() 
        ,data.loc['2015-11-1' : '2016-02-29'].to_numpy()
        ,data.loc['2016-03-1' : '2016-06-30'].to_numpy()
        ,data.loc['2016-07-1' : '2016-10-30'].to_numpy()
        ,data.loc['2016-11-1' : '2017-02-28'].to_numpy()
        ,data.loc['2017-03-1' : '2017-06-30'].to_numpy()
        ,data.loc['2017-07-1' : '2017-10-31'].to_numpy()
        ,data.loc['2017-11-1' : '2018-02-28'].to_numpy()
        ,data.loc['2018-03-1' : '2018-06-28'].to_numpy()
        ,data.loc['2018-07-1' : '2018-10-31'].to_numpy()
        ,data.loc['2018-11-1' : '2019-02-28'].to_numpy()
        ,data.loc['2019-03-1' : '2019-06-28'].to_numpy()
        ,data.loc['2019-07-1' : '2019-10-31'].to_numpy()
        ,data.loc['2019-11-1' : '2020-02-28'].to_numpy()
        ,data.loc['2020-03-1' : '2020-06-28'].to_numpy()
        ,data.loc['2020-07-1' : '2020-10-31'].to_numpy()] 

    #print_stats(data_split[0])
    # decompose(data)
    autocorr(data)

if __name__ == '__main__':
    main()
