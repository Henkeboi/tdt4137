import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize
from scipy import stats

from holt_winters import HoltWinters
from holt_winters_without_leap_year import HoltWintersWithoutLeapYear
from hybrid import Hybrid

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

def autocorr(x, t):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

def cross_validation():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.

    data_without_leap_year = pd.read_csv('data/Demand_for_California_hourly_UTC_time_without_leap_year.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data_without_leap_year = data_without_leap_year.reindex(index=data_without_leap_year.index[::-1])

    horizon = 7 * 24
    time_series = [data.loc['2016-01-01' : '2019-08-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-08-07'].to_numpy()
        ,data.loc['2016-01-01' : '2019-08-14'].to_numpy()
        ,data.loc['2016-01-01' : '2019-08-21'].to_numpy()
        ,data.loc['2016-01-01' : '2019-08-28'].to_numpy()
        ,data.loc['2016-01-01' : '2019-09-07'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-14'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-21'].to_numpy()
        ,data.loc['2016-01-01' : '2019-11-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-11-07'].to_numpy()]

    time_series_without_leap_year = [data_without_leap_year.loc['2016-01-01' : '2019-08-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-08-07'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-08-14'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-08-21'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-08-28'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-09-07'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-10-14'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-10-21'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-11-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-11-07'].to_numpy()]

    sum_mae_error_add = 0
    sum_mae_error_mul = 0
    sum_mae_error_week = 0
    sum_mae_error_year = 0
    sum_mae_error_year_without_leap_year = 0
    sum_mae_error_hybrid = 0

    do_training = False
    for i in range(0, len(time_series)):
        print("Iteration: " + str(i))

        ts = time_series[i]
        hw = HoltWinters(ts[:-horizon], horizon)

        forecast_additive, residuals_additive = hw.holt_winters_additive_predict(do_training)
        mae_error_add = mae(ts[-horizon:], forecast_additive)
        sum_mae_error_add = sum_mae_error_add + mae_error_add 
        print("Additive:" + str(mae_error_add))

        forecast_multiplicative, residuals_multiplicative = hw.holt_winters_multiplicative_predict(do_training)
        mae_error_mul = mae(ts[-horizon:], forecast_multiplicative)
        sum_mae_error_mul = sum_mae_error_mul + mae_error_mul
        print("Multiplicative:" + str(mae_error_mul))

        forecast_week_extended, residuals_week = hw.holt_winters_multiplicative_week_extended_predict(do_training)
        mae_error_week = mae(ts[-horizon:], forecast_week_extended)
        sum_mae_error_week = sum_mae_error_week + mae_error_week
        print("Extended week:" + str(mae_error_week))

        forecast_year_extended, residuals_year = hw.holt_winters_multiplicative_year_extended_predict(do_training)
        mae_error_year = mae(ts[-horizon:], forecast_year_extended)
        sum_mae_error_year = sum_mae_error_year + mae_error_year
        print("Extended year:" + str(mae_error_year))

        ts_without_leap_year = time_series_without_leap_year[i]
        hw_without_leap_year = HoltWintersWithoutLeapYear(ts_without_leap_year[:-horizon], horizon)
        forecast_year_extended_without_leap_year, residuals_year_without_leap_year = hw_without_leap_year.holt_winters_multiplicative_year_extended_predict(do_training)
        mae_error_year_without_leap_year = mae(ts_without_leap_year[-horizon:], forecast_year_extended_without_leap_year)
        sum_mae_error_year_without_leap_year = sum_mae_error_year_without_leap_year + mae_error_year_without_leap_year
        print("Extended year without leap:" + str(mae_error_year_without_leap_year))

        hybrid = Hybrid(ts[:-horizon], horizon)
        forecast_hybrid = hybrid.forecast()
        mae_error_hybrid = mae(ts[-horizon:], forecast_hybrid)
        sum_mae_error_hybrid = sum_mae_error_hybrid + mae_error_hybrid
        print("Hybrid:" + str(mae_error_hybrid))

         
    print("Sum additive:" + str(sum_mae_error_add))
    print("Sum multiplicative:" + str(sum_mae_error_mul))
    print("Sum extended week:" + str(sum_mae_error_week))
    print("Sum extended year:" + str(sum_mae_error_year))
    print("Sum extended year without leap:" + str(sum_mae_error_year_without_leap_year))
    print("Sum hybrid:" + str(sum_mae_error_hybrid))

def main():
    cross_validation()

    
   
if __name__ == '__main__':
    main()
