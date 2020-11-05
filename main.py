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
    for i in range(100, len(ts)):
        step_error = pow(ts[i] - forecast[i], 2)
        error.append(step_error)
    #plt.plot(error)
    #plt.show()
    return np.sum(error)

def holt_winters_additive(ts, horizon):
    alpha = 0.7    # 0 <= alpha <= 1
    beta = 0.5     # 0 <= beta <= 1
    gamma = 0.001 #1 - alpha    # 0 <= gamma <= 1 - alpha
    h = 1
    m = 24
    k = math.floor((h - 1.0) / float(m))
    prev_level = 10000
    prev_trend = 10000
    forecast = np.full(len(ts) + horizon, math.nan)
    seasonal_components = np.full(len(ts) + horizon, math.nan)

    level_eq = lambda alpha, ts, prev_level, prev_trend, prev_season : alpha * (ts[0] - prev_season) + (1 - alpha) * (prev_level + prev_trend)
    trend_eq = lambda beta, level, prev_level, prev_trend : beta * (level - prev_level) + (1 - beta) * prev_trend
    season_eq = lambda gamma, ts, prev_level, prev_trend, prev_season : gamma * (ts[0] - prev_level - prev_trend) + (1 - gamma) * prev_season

    forecast[1] = ts[0]
    for i in range(2, len(ts) + 1):
        # Calculate components
        if i - m >= 0:
            level = level_eq(alpha, ts[i - 1], prev_level, prev_trend, seasonal_components[i - m]) # Add seasonality
            seasonal_components[i - 2] = season_eq(gamma, ts[i - 1], prev_level, prev_trend, seasonal_components[i - m]) 
        else: 
            seasonal_components[i - 2] = 0
            level = level_eq(alpha, ts[i - 1], prev_level, prev_trend, 0)
        trend = trend_eq(beta, level, prev_level, prev_trend)

        # Make forecasts
        if (i - m * (k + 1) < 0):
            forecast[i] = level + h * trend 
        else:
            forecast[i] = level + h * trend + seasonal_components[i - m * (k + 1)]
        prev_level = level
        prev_trend = trend

    forecast[len(ts) + 1 :] = forecast[len(ts)] # Make prediction
    return forecast[1:]

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.

    horizon = 100
    ts = data.loc['2019-1-1' : '2019-3-31'].to_numpy() # 

    hw = HoltWinters(ts[:-horizon], horizon)
    forecast_multiplicative = hw.holt_winters_multiplicative_predict()
    forecast_season_extended = hw.holt_winters_multiplicative_seasonality_extended_predict()

    x = np.arange(0, horizon, 1)
    plt.plot(x, ts[-horizon:], 'b')
    plt.plot(x, forecast_multiplicative, 'r')
    plt.plot(x, forecast_season_extended, 'y')
    plt.show()

if __name__ == '__main__':
    main()
