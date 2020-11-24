import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize
from scipy import stats
from sklearn.metrics import mean_squared_error
import seaborn as sns

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

#def mae(ts, forecast):
#    assert(len(ts) == len(forecast))
#    error = 0
#    for i in range(0, len(ts)):
#        error = error + abs(ts[i] - forecast[i])
#    return error / len(ts)


def mae(ts, forecast): #rmse
    assert(len(ts) == len(forecast))
    return math.sqrt(mean_squared_error(ts, forecast))

def autocorr(x, t):
    return np.corrcoef(np.array([x[:-t], x[t:]]))

def cross_validation():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.

    data_without_leap_year = pd.read_csv('data/Demand_for_California_hourly_UTC_time_without_leap_year.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data_without_leap_year = data_without_leap_year.reindex(index=data_without_leap_year.index[::-1])

    horizon = 31 * 24
    time_series = [data.loc['2016-01-01' : '2019-02-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-03-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-04-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-05-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-06-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-07-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-08-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-09-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-10-01'].to_numpy()
        ,data.loc['2016-01-01' : '2019-11-01'].to_numpy()]


    time_series_without_leap_year = [data_without_leap_year.loc['2016-01-01' : '2019-02-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-03-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-04-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-05-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-06-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-07-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-08-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-09-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-10-01'].to_numpy()
        ,data_without_leap_year.loc['2016-01-01' : '2019-11-01'].to_numpy()]

    sum_mae_error_add = 0
    sum_mae_error_mul = 0
    sum_mae_error_week = 0
    sum_mae_error_year = 0
    sum_mae_error_year_without_leap_year = 0
    sum_mae_error_hybrid = 0

    do_training = False
    for i in range(0, len(time_series)):
        #print("Iteration: " + str(i))

        ts = time_series[i]
        hw = HoltWinters(ts[:-horizon], horizon)
        x = np.linspace(0, horizon, horizon)

        forecast_additive = hw.holt_winters_additive_predict(do_training)[0]
        residuals_additive = ts[-horizon:] - forecast_additive
        mae_error_add = mae(ts[-horizon:], forecast_additive)
        sum_mae_error_add = sum_mae_error_add + mae_error_add 
        sns.lineplot(x, forecast_additive.flatten())
        sns.lineplot(x, ts[-horizon:].flatten())
        plt.show()
        print("Additive:" + str(mae_error_add))

        #forecast_multiplicative, residuals_multiplicative = hw.holt_winters_multiplicative_predict(do_training)
        #residuals_multiplicative = ts[-horizon:] - forecast_multiplicative
        #mae_error_mul = mae(ts[-horizon:], forecast_multiplicative)
        #sum_mae_error_mul = sum_mae_error_mul + mae_error_mul
        #sns.lineplot(x, forecast_multiplicative.flatten())
        #sns.lineplot(x, ts[-horizon:].flatten())
        #plt.show()
        #print("Multiplicative:" + str(mae_error_mul))

        #forecast_week_extended = hw.holt_winters_multiplicative_week_extended_predict(do_training)[0]
        #residuals_week = ts[-horizon:] - forecast_week_extended
        #mae_error_week = mae(ts[-horizon:], forecast_week_extended)
        #sum_mae_error_week = sum_mae_error_week + mae_error_week
        #sns.lineplot(x, forecast_week_extended.flatten())
        #sns.lineplot(x, ts[-horizon:].flatten())
        #plt.show()
        #print("Extended week:" + str(mae_error_week))

        #forecast_year_extended = hw.holt_winters_multiplicative_year_extended_predict(do_training)[0]
        #residuals_year = ts[-horizon:] - forecast_year_extended
        #mae_error_year = mae(ts[-horizon:], forecast_year_extended)
        #sum_mae_error_year = sum_mae_error_year + mae_error_year
        #sns.lineplot(x, forecast_year_extended.flatten())
        #sns.lineplot(x, ts[-horizon:].flatten())
        #plt.show()
        #print("Extended year:" + str(mae_error_year))

        #ts_without_leap_year = time_series_without_leap_year[i]
        #hw_without_leap_year = HoltWintersWithoutLeapYear(ts_without_leap_year[:-horizon], horizon)
        #forecast_year_extended_without_leap_year = hw_without_leap_year.holt_winters_multiplicative_year_extended_predict(do_training)[0]
        #residuals_without_leap_year = ts[-horizon:] - forecast_year_extended_without_leap_year
        #mae_error_year_without_leap_year = mae(ts_without_leap_year[-horizon:], forecast_year_extended_without_leap_year)
        #sum_mae_error_year_without_leap_year = sum_mae_error_year_without_leap_year + mae_error_year_without_leap_year
        #sns.lineplot(x, forecast_year_extended_without_leap_year.flatten())
        #sns.lineplot(x, ts[-horizon:].flatten())
        #plt.show()
        #print("Extended year without leap:" + str(mae_error_year_without_leap_year))

        #hybrid = Hybrid(ts[:-horizon], horizon)
        #forecast_hybrid = hybrid.forecast()
        #residuals_hybrid = ts[-horizon:] - forecast_hybrid
        #mae_error_hybrid = mae(ts[-horizon:], forecast_hybrid)
        #sum_mae_error_hybrid = sum_mae_error_hybrid + mae_error_hybrid
        #sns.lineplot(x, forecast_hybrid.flatten())
        #sns.lineplot(x, ts[-horizon:].flatten())
        #plt.show()
        ##print("Hybrid:" + str(mae_error_hybrid))
        #sns.lineplot(forecast_hybrid)
        quit()

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
