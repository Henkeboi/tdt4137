import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize

class HoltWinters:
    def __init__(self):
        data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
        data = data.reindex(index=data.index[::-1])
        data.index.freq = 'H' # Hourly data.
        self.ts = data.to_numpy() 

    def sse(self, forecast):
        assert(len(self.ts) == len(forecast))
        error = []
        for i in range(0, len(self.ts)):
            step_error = pow(self.ts[i] - forecast[i], 2)
            error.append(step_error)
        return np.sum(error)

    def moving_average(self, index, index_range):
        avg = 0
        i = index - index_range
        if i < 0:
            i = 0
        num_elements = 0
        while i <= index + index_range:
            avg = avg + self.ts[i]
            i = i + 1
            num_elements = num_elements + 1
        return avg / num_elements


    def holt_winters_multiplicative_seasonality_extended(self, x):
        alpha = x[0]
        beta = x[1]
        gamma_day = x[2]
        gamma_week = x[3]
        k = 1
        day_period_length = 24
        week_period_length = 7 * 24
        forecast = np.full(len(self.ts) + k, math.nan)
        day_seasonality = np.full(len(self.ts), math.nan)
        week_seasonality = np.full(len(self.ts), math.nan)

        level_eq = lambda alpha, observation, M_0, level_prev, trend_prev : alpha * observation / M_0 + (1 - alpha) * (level_prev + trend_prev)
        trend_eq = lambda beta, level, level_prev, trend_prev : beta * (level - level_prev) + (1 - beta) * trend_prev
        seasonality_eq = lambda gamma, observation, seasonality_prev, level, M_0 : \
            gamma * observation * seasonality_prev / (level * M_0) + (1 - gamma) * seasonality_prev

        level_initial = np.average(self.ts[0 : 2 * day_period_length])
        trend_initial = abs(np.average(self.ts[0 : day_period_length]) - np.average(self.ts[day_period_length : 2 * day_period_length])) / day_period_length
        for i in range(0, len(self.ts)):
            # Update smoothing equations 
            if i > 3 * week_period_length:
                M_0 = day_seasonality[i - day_period_length] * week_seasonality[i - week_period_length]
                level = level_eq(alpha, self.ts[i], M_0, level_prev, trend_prev)
                trend = trend_eq(beta, level, level_prev, trend_prev)
                day_seasonality[i] = seasonality_eq(gamma_day, self.ts[i], day_seasonality[i - day_period_length], level, M_0)
                week_seasonality[i] = seasonality_eq(gamma_week, self.ts[i], week_seasonality[i - week_period_length], level, M_0)
            elif i > 3 * day_period_length:
                M_0 = day_seasonality[i - day_period_length] 
                day_seasonality[i] = seasonality_eq(gamma_day, self.ts[i], day_seasonality[i - day_period_length], level, M_0)
                week_seasonality[i] = \
                    (self.ts[i] / self.moving_average(i, week_period_length) + self.ts[i + week_period_length] / self.moving_average(i + week_period_length, week_period_length)) / 2
            else:
                level = level_initial
                trend = trend_initial
                prev_seasonality = 1
                day_seasonality[i] = \
                    (self.ts[i] / self.moving_average(i, day_period_length) + self.ts[i + day_period_length] / self.moving_average(i + day_period_length, day_period_length)) / 2
                week_seasonality[i] = \
                    (self.ts[i] / self.moving_average(i, week_period_length) + self.ts[i + week_period_length] / self.moving_average(i + week_period_length, week_period_length)) / 2
     
            # Smooth 
            if i > 3 *  week_period_length:
                M_0 = day_seasonality[i - day_period_length] * week_seasonality[i - week_period_length]
                forecast[i] = (level + trend) * M_0
            elif i > 3 * day_period_length:
                forecast[i] = self.ts[i]
            else:
                forecast[i] = self.ts[i]
 
            # Store values    
            level_prev = level
            trend_prev = trend

        plot = False
        if plot == True:
            self.ts = np.append(self.ts, [math.nan] * k) # Make equal length
            self.ts = self.ts[3 * week_period_length :]
            forecast = forecast[3 * week_period_length :]
            data_frame = pd.DataFrame.from_dict({"Data" : self.ts, "Forecast" : forecast})
            data_frame.plot()
            plt.show()
            return (forecast[:-1], self.ts[:-1])
        return self.sse(forecast[:-1])
        #return forecast[:-1], self.ts, self.sse(forecast[:-1])

    def optimize_holt_winters_multiplicative_seasonality_extended(self):
        alpha = 0.999             # 0 <= alpha <= 1
        beta = 0.0001             # 0 <= beta <= 1
        gamma_day = 0.9        # 0 <= gamma_day <= 1
        gamma_week = 0.7

        initial_guess = np.array([alpha, beta, gamma_day, gamma_week])
        bound_on_variables = ((0, 1), (0, 1), (0, 1), (0, 1))
        tolerance = 0.9
        variables_minimized = minimize(self.holt_winters_multiplicative_seasonality_extended, initial_guess, bounds=bound_on_variables, tol=tolerance).x
        print(variables_minimized)
        print(self.holt_winters_multiplicative_seasonality_extended(variables_minimized))

        

        
