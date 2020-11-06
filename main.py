import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize

from holt_winters import HoltWinters

def sse(ts, forecast):
    assert(len(ts) == len(forecast))
    error = []
    for i in range(0, len(ts)):
        step_error = pow(ts[i] - forecast[i], 2)
        error.append(step_error)
    return np.sum(error)

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.

    horizon = 100
    ts = data.loc['2019-1-1' : '2019-3-31'].to_numpy() 

    hw = HoltWinters(ts[:-horizon], horizon)
    forecast_multiplicative = hw.holt_winters_multiplicative_predict()
    forecast_additive = hw.holt_winters_additive_predict()
    forecast_season_extended = hw.holt_winters_multiplicative_seasonality_extended_predict()

    print(sse(ts[-horizon:], forecast_additive))
    print(sse(ts[-horizon:], forecast_multiplicative))
    print(sse(ts[-horizon:], forecast_season_extended))

    #x = np.arange(0, horizon, 1)
    #plt.plot(x, ts[-horizon:], 'r')
    #plt.plot(x, forecast_additive, 'r')
    #plt.plot(x, forecast_multiplicative, 'r')
    #plt.plot(x, forecast_season_extended, 'y')
    #plt.show()

if __name__ == '__main__':
    main()
