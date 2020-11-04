import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

def sse(ts, forecast):
    assert(len(ts) == len(forecast))
    error = []
    for i in range(100, len(ts)):
        step_error = pow(ts[i][0] - forecast[i], 2)
        error.append(step_error)
    #plt.plot(error)
    #plt.show()
    return sum(error)

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
    #ts = np.append(ts, [math.nan] * horizon) # Make equal length
    #data_frame = pd.DataFrame.from_dict({"Data" : ts, "Forecast" : forecast})
    #data_frame.plot()
    #plt.show()
    return forecast[1:]

def holt_winters_multiplicative(ts, horizon):
    alpha = 0.7     # 0 <= alpha <= 1
    beta = 0.1      # 0 <= beta <= 1
    gamma = 0.25     # 0 <= gamma <= 1
    h = 1
    m = 24
    k = math.floor((h - 1.0) / float(m))
    prev_level = 10e3
    prev_trend = 10e4
    forecast = np.full(len(ts) + horizon, math.nan)
    seasonal_components = np.full(len(ts) + horizon, math.nan)

    level_eq = lambda alpha, ts, prev_level, prev_trend, prev_season : alpha * (ts[0] - prev_season) + (1 - alpha) * (prev_level + prev_trend)
    trend_eq = lambda beta, level, prev_level, prev_trend : beta * (level - prev_level) + (1 - beta) * prev_trend
    season_eq = lambda gamma, ts, prev_level, prev_trend, prev_season : gamma * ts[0] / (prev_level - prev_trend) + (1 - gamma) * prev_season

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
            forecast[i] = (level + h * trend) * seasonal_components[i - m * (k + 1)]

        prev_level = level
        prev_trend = trend

    forecast[len(ts) + 1 :] = forecast[len(ts)] # Make prediction
    #ts = np.append(ts, [math.nan] * horizon) # Make equal length
    #data_frame = pd.DataFrame.from_dict({"Data" : ts, "Forecast" : forecast})
    #data_frame.plot()
    #plt.show()
    return forecast[1:]

def holt_winters_multiplicative_seasonality_extended(ts):
    alpha = 0.77             # 0 <= alpha <= 1
    beta = 0.02             # 0 <= beta <= 1
    gamma_day = 0.99        # 0 <= gamma_day <= 1
    gamma_week = 0.99
    k = 1
    day_period_length = 24
    week_period_length = 7 * 24
    forecast = np.full(len(ts) + k, math.nan)
    day_seasonality = np.full(len(ts) - day_period_length, math.nan)
    week_seasonality = np.full(len(ts) - week_period_length, math.nan)

    level_eq = lambda alpha, observation, M_0, level_prev, trend_prev : alpha * observation / M_0 + (1 - alpha) * (level_prev + trend_prev)
    trend_eq = lambda beta, level, level_prev, trend_prev : beta * (level - level_prev) + (1 - beta) * trend_prev

    seasonality_eq = lambda gamma, observation, seasonality_prev, level, M_0 : \
            gamma * observation * seasonality_prev / (level + M_0) + (1 - gamma) * seasonality_prev

    level_prev = 10e5 # todo: Correct the initial value
    trend_prev = 10  # todo: Correct the initial value
    for i in range(0, len(ts)):
        # Update smoothing equations 
        if i > week_period_length * 200:
            M_0 = day_seasonality[i - day_period_length - 1] #* week_seasonality[i - week_period_length - 1]
            #level = level_eq(alpha, ts[i], M_0, level_prev, trend_prev) 
            level = level_eq(alpha, ts[i], 1, level_prev, trend_prev) 
            day_seasonality[i - day_period_length] = seasonality_eq(gamma_day, ts[i], day_seasonality[i - day_period_length - 1], level, M_0)
            week_seasonality[i - week_period_length] = 1 #seasonality_eq(gamma_week, ts[i], week_seasonality[i - week_period_length - 1], level, M_0)
            #print(week_seasonality[i - week_period_length])
            #print(day_seasonality[i - day_period_length])
        elif i > day_period_length * 200:
            M_0 = day_seasonality[i - day_period_length - 1]
            #level = level_eq(alpha, ts[i], M_0, level_prev, trend_prev) # Todo: Using 1 as day seasonality init val
            level = level_eq(alpha, ts[i], 1, level_prev, trend_prev) # Todo: Using 1 as day seasonality init val
            # Todo: Calculate init val
            day_seasonality[i - day_period_length] = seasonality_eq(gamma_day, ts[i], day_seasonality[i - day_period_length - 1], level, M_0)
            week_seasonality[i - week_period_length] = 1
        else:
            level = level_eq(alpha, ts[i], 1, level_prev, trend_prev) # Todo: Using 1 as day seasonality init val
            day_seasonality[i - day_period_length] = seasonality_eq(gamma_day, ts[i], 1, level, 1)
            week_seasonality[i - week_period_length] = 1
            trend = trend_eq(beta, level, level_prev, trend_prev)

        # Smooth 
        if i > week_period_length: 
            forecast[i] = (level + trend) * day_seasonality[i - day_period_length - 1] #* week_seasonality[i - week_period_length - 1]
        elif i > day_period_length:
            forecast[i] = (level + trend) * day_seasonality[i - day_period_length - 1]
        else:
            forecast[i] = level + trend
        
        # Store values    
        level_prev = level
        trend_prev = trend

    #ts = np.append(ts, [math.nan] * k) # Make equal length
    #data_frame = pd.DataFrame.from_dict({"Data" : ts, "Forecast" : forecast})
    #data_frame.plot()
    #plt.show()
    return forecast[:-1]

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    ts = data.to_numpy()
    #ts = np.true_divide(ts, 10e6) # Convert from Megawatt hours to watt hours

    #forecast = holt_winters_additive(copy.deepcopy(ts), 1) 
    #print(sse(ts, forecast))
    #forecast = holt_winters_multiplicative(copy.deepcopy(ts), 1) 
    #print(sse(ts, forecast))
    forecast = holt_winters_multiplicative_seasonality_extended(copy.deepcopy(ts)) 
    print(sse(ts, forecast))

    #[rest_ts, level_ts] = level_eq(ts, alpha, horizon)

if __name__ == '__main__':
    main()
