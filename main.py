import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

def sse(ts, forecast):
    assert(len(ts) == len(forecast))
    error = 0
    for i in range(0, len(ts)):
        step_error = pow(ts[i][0] - forecast[i], 2)
        #if not math.isnan(step_error):
        error = step_error
    return error

def level_eq(ts, alpha, horizon):
    level_ts = np.full(len(ts) + horizon, math.nan)
    level_ts[1] = ts[0]
    for i in range(2, len(ts) + 1):
        level_ts[i] = alpha * ts[i - 1] + (1 - alpha) * level_ts[i - 1]

    level_ts[len(ts) + 1 :] = level_ts[len(ts)] # Make prediction
    ts = np.append(ts, [math.nan] * horizon) # Make equal length

    #data_frame = pd.DataFrame.from_dict({"Data" : ts, "Forecast" : level_ts, "Error" : ts - level_ts})
    #data_frame.plot()
    #plt.show()
    return (ts[:-1] - level_ts[1:], level_ts[1:])


def holt_winters_additive(ts, horizon):
    alpha = 0.9    # 0 <= alpha <= 1
    beta = 0.5     # 0 <= beta <= 1
    gamma = 0.01#1 - alpha    # 0 <= gamma <= 1 - alpha
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
        level = level_eq(alpha, ts[i - 1], prev_level, prev_trend, 0) # Add seasonality
        trend = trend_eq(beta, level, prev_level, prev_trend)
        if i - m >= 0:
            seasonal_components[i - 2] = season_eq(gamma, ts[i - 1], prev_level, prev_trend, seasonal_components[i - m]) # Prev season should be i - m + 1
        else:
            seasonal_components[i - 2] = 0

        if (i - m * (k + 1) < 0):
            forecast[i] = level + h * trend 
        else:
            forecast[i] = level + h * trend + seasonal_components[i - m * (k + 1)]
        prev_level = level
        prev_trend = trend

    forecast[len(ts) + 1 :] = forecast[len(ts)] # Make prediction
    ts = np.append(ts, [math.nan] * horizon) # Make equal length
    data_frame = pd.DataFrame.from_dict({"Data" : ts, "Forecast" : forecast, "Error" : ts - forecast})
    data_frame.plot()
    plt.show()
    return forecast[1:]

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    ts = data.to_numpy()
    #ts = np.true_divide(ts, 10e6) # Convert from Megawatt hours to watt hours

    horizon = 1
    forecast = holt_winters_additive(copy.deepcopy(ts), horizon) 
    print(sse(ts, forecast))

    #[rest_ts, level_ts] = level_eq(ts, alpha, horizon)

if __name__ == '__main__':
    main()
