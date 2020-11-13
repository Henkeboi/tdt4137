import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize
from scipy import stats

from holt_winters import HoltWinters

def sse(ts, forecast):
    assert(len(ts) == len(forecast))
    error = 0
    for i in range(0, len(ts)):
        step_error = pow(ts[i] - forecast[i], 2)
        error = error + step_error
    return error

def mae(ts, forecast):
    assert(len(ts) == len(forecast))
    error = 0
    for i in range(0, len(ts)):
        error = error + abs(ts[i] - forecast[i])
    return error / len(ts)

def mse(ts, forecast):
    pass

def mase(ts, forecast, ts_index_start): # Seasonal version
    season_length = 24
    error = 0
    for i in range(0, len(ts)):
        abs_error = abs(ts[i] - forecast[i])
        abs_seasonal_diff = abs(ts[ts_index_start + i + 1] - ts[ts_index_start + i])
        error = error + abs_error / abs_seasonal_diff

def experiment_1():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    horizon = 24
    
    time_series = [data.loc['2015-07-1' : '2015-10-31'].to_numpy() 
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

    i = 0
    sum_mae_error_add = 0
    sum_mae_error_mul = 0
    sum_mae_error_mse = 0
    do_training = False
    for ts in time_series:
        hw = HoltWinters(ts[:-horizon], horizon)
        forecast_additive = hw.holt_winters_additive_predict(do_training)
        forecast_multiplicative = hw.holt_winters_multiplicative_predict(do_training)
        forecast_week_extended = hw.holt_winters_multiplicative_week_extended_predict(do_training)

        print("Mean squared error add: " + str(mae(ts[-horizon:], forecast_additive)))
        print("Mean squared error mul: " + str(mae(ts[-horizon:], forecast_multiplicative)))
        print("Mean squared error mse: " + str(mae(ts[-horizon:], forecast_week_extended)))

        sum_mae_error_add = sum_mae_error_add + mae(ts[-horizon:], forecast_additive)
        sum_mae_error_mul = sum_mae_error_mul + mae(ts[-horizon:], forecast_multiplicative)
        sum_mae_error_mse = sum_mae_error_mse + mae(ts[-horizon:], forecast_week_extended)

    print(sum_mae_error_add)
    print(sum_mae_error_mul)
    print(sum_mae_error_mse)

def experiment_2():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    horizon = 7 * 24
    
    time_series = [data.loc['2016-01-01' : '2019-10-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-02'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-03'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-04'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-05'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-06'].to_numpy()]

    i = 0
    sum_mae_error_add = 0
    sum_mae_error_mul = 0
    sum_mae_error_mse = 0
    sum_mae_error_year = 0
    do_training = False
    for ts in time_series:
        hw = HoltWinters(ts[:-horizon], horizon)
        forecast_additive = hw.holt_winters_additive_predict(do_training)
        forecast_multiplicative = hw.holt_winters_multiplicative_predict(do_training)
        forecast_week_extended = hw.holt_winters_multiplicative_week_extended_predict(do_training)
        forecast_year_extended = hw.holt_winters_multiplicative_year_extended_predict(do_training)

        print("Mean squared error add: " + str(mae(ts[-horizon:], forecast_additive)))
        print("Mean squared error mul: " + str(mae(ts[-horizon:], forecast_multiplicative)))
        print("Mean squared error week: " + str(mae(ts[-horizon:], forecast_week_extended)))
        print("Mean squared error year: " + str(mae(ts[-horizon:], forecast_year_extended)))

        sum_mae_error_add = sum_mae_error_add + mae(ts[-horizon:], forecast_additive)
        sum_mae_error_mul = sum_mae_error_mul + mae(ts[-horizon:], forecast_multiplicative)
        sum_mae_error_mse = sum_mae_error_mse + mae(ts[-horizon:], forecast_week_extended)
        sum_mae_error_year = sum_mae_error_year + mae(ts[-horizon:], forecast_year_extended)

    print(sum_mae_error_add)
    print(sum_mae_error_mul)
    print(sum_mae_error_mse)
    print(sum_mae_error_year)


def main():
    #experiment_1()
    experiment_2()
    #x = np.arange(0, horizon, 1)
    #plt.plot(x, ts[-horizon:], 'r')
    #plt.plot(x, forecast_additive, 'g')
    #plt.plot(x, forecast_multiplicative, 'b')
    #plt.plot(x, forecast_season_extended, 'y')
    #plt.show()

if __name__ == '__main__':
    main()
