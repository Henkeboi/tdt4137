import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize

class HoltWinters:
    def __init__(self, ts, horizon):
        self.ts = ts
        self.horizon = horizon
        self.is_training = True
        #self.year_seasonality = np.full(len(self.ts) + self.horizon, math.nan)
        self.year_seasonality = np.load('variables/year_array.npy')
        while len(self.year_seasonality) != len(ts) + horizon:
            self.year_seasonality = np.append(self.year_seasonality, math.nan)

    def sse(self, forecast):
        assert(len(self.ts) == len(forecast))
        error = []
        for i in range(0, len(self.ts)): # Shave off the first values that often have huge errors
            step_error = pow(self.ts[i] - forecast[i], 2)
            error.append(step_error)
        return np.sum(error)

    def moving_average(self, index, index_range):
        avg = 0
        i = index - index_range
        if i < 0:
            i = 0
        num_elements = 0
        while i <= index + index_range and i < len(self.ts):
            avg = avg + self.ts[i]
            i = i + 1
            num_elements = num_elements + 1
        return avg / num_elements

    def holt_winters_multiplicative_week_extended(self, x, return_error=True):
        alpha = x[0]
        beta = x[1]
        gamma_day = x[2]
        gamma_week = x[3]
        day_period_length = 24
        week_period_length = 7 * 24
        if self.is_training:
            forecast = np.full(len(self.ts), math.nan)
        else:
            forecast = np.full(len(self.ts) + self.horizon, math.nan)
        day_seasonality = np.full(len(self.ts) + self.horizon, math.nan)
        week_seasonality = np.full(len(self.ts) + self.horizon, math.nan)

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
        
        if not self.is_training: 
            # Predict
            forecast = np.full(self.horizon, math.nan)
            for i in range(len(self.ts), len(self.ts) + self.horizon):
                M_0 = day_seasonality[i - day_period_length] * week_seasonality[i - week_period_length]
                trend = trend_eq(beta, level, level_prev, trend_prev)
                if i == len(self.ts):
                    level = level_eq(alpha, self.ts[-1], M_0, level_prev, trend_prev)
                else:
                    level = level_eq(alpha, forecast[i - len(self.ts) - 1], M_0, level_prev, trend_prev)
                forecast[i - len(self.ts)] = (level + trend) * M_0
                day_seasonality[i] = seasonality_eq(gamma_day, forecast[i - len(self.ts)], day_seasonality[i - day_period_length], level, M_0)
                week_seasonality[i] = seasonality_eq(gamma_week, forecast[i - len(self.ts)], week_seasonality[i - week_period_length], level, M_0)
            return forecast
        else:
            if return_error:
                return self.sse(forecast)
            else:
                return forecast

    def holt_winters_multiplicative_week_extended_predict(self, do_training):
        alpha = 0.9             # 0 <= alpha <= 1
        beta = 0.0001           # 0 <= beta <= 1
        gamma_day = 0.9         # 0 <= gamma_day <= 1
        gamma_week = 0.7        # 0 <= gamma_week <= 1

        initial_guess = np.array([alpha, beta, gamma_day, gamma_week])
        bound_on_variables = ((0, 1), (0, 1), (0, 1), (0, 1))
        tolerance = 0.3
        variables_optimized = initial_guess
        self.is_training = True
        if do_training:
            variables_optimized = minimize(self.holt_winters_multiplicative_week_extended, initial_guess, bounds=bound_on_variables, tol=tolerance).x
        residuals = np.array([])
        smoothed_series = self.holt_winters_multiplicative_week_extended(variables_optimized, False)
        for i in range(0, len(self.ts)):
            residuals = np.append(residuals, self.ts[i][0] - smoothed_series[i])

        self.is_training = False
        return self.holt_winters_multiplicative_week_extended(variables_optimized), residuals

    def holt_winters_multiplicative_year_extended(self, x, return_error=True):
        alpha = x[0]
        beta = x[1]
        gamma_day = x[2]
        gamma_week = x[3]
        gamma_year = x[4]
        
        day_period_length = 24
        week_period_length = 7 * day_period_length
        year_period_length = 52 * week_period_length
        if self.is_training:
            forecast = np.full(len(self.ts), math.nan)
        else:
            forecast = np.full(len(self.ts) + self.horizon, math.nan)
        day_seasonality = np.full(len(self.ts) + self.horizon, math.nan)
        week_seasonality = np.full(len(self.ts) + self.horizon, math.nan)

        level_eq = lambda alpha, observation, M_0, level_prev, trend_prev : alpha * observation / M_0 + (1 - alpha) * (level_prev + trend_prev)
        trend_eq = lambda beta, level, level_prev, trend_prev : beta * (level - level_prev) + (1 - beta) * trend_prev
        seasonality_eq = lambda gamma, observation, seasonality_prev, level, M_0 : \
            gamma * observation * seasonality_prev / (level * M_0) + (1 - gamma) * seasonality_prev

        level_initial = np.average(self.ts[0 : 2 * day_period_length])
        trend_initial = abs(np.average(self.ts[0 : day_period_length]) - np.average(self.ts[day_period_length : 2 * day_period_length])) / day_period_length

        for i in range(0, len(self.ts)):
            # Update smoothing equations 
            if i > 2 * year_period_length:
                M_0 = day_seasonality[i - day_period_length] * week_seasonality[i - week_period_length] * self.year_seasonality[i - year_period_length]
                level = level_eq(alpha, self.ts[i], M_0, level_prev, trend_prev)
                trend = trend_eq(beta, level, level_prev, trend_prev)
                day_seasonality[i] = seasonality_eq(gamma_day, self.ts[i], day_seasonality[i - day_period_length], level, M_0)
                week_seasonality[i] = seasonality_eq(gamma_week, self.ts[i], week_seasonality[i - week_period_length], level, M_0)
                self.year_seasonality[i] = seasonality_eq(gamma_year, self.ts[i], self.year_seasonality[i - year_period_length], level, M_0)
            elif i > 3 * week_period_length:
                M_0 = day_seasonality[i - day_period_length] * week_seasonality[i - week_period_length]
                level = level_eq(alpha, self.ts[i], M_0, level_prev, trend_prev)
                trend = trend_eq(beta, level, level_prev, trend_prev)
                day_seasonality[i] = seasonality_eq(gamma_day, self.ts[i], day_seasonality[i - day_period_length], level, M_0)
                week_seasonality[i] = seasonality_eq(gamma_week, self.ts[i], week_seasonality[i - week_period_length], level, M_0)
                if math.isnan(self.year_seasonality[i]):
                    self.year_seasonality[i] =  \
                        (self.ts[i] / self.moving_average(i, year_period_length) + self.ts[i + year_period_length] / self.moving_average(i + year_period_length, year_period_length)) / 2
            elif i > 3 * day_period_length:
                M_0 = day_seasonality[i - day_period_length] 
                day_seasonality[i] = seasonality_eq(gamma_day, self.ts[i], day_seasonality[i - day_period_length], level, M_0)
                week_seasonality[i] = \
                    (self.ts[i] / self.moving_average(i, week_period_length) + self.ts[i + week_period_length] / self.moving_average(i + week_period_length, week_period_length)) / 2
                if math.isnan(self.year_seasonality[i]):
                    self.year_seasonality[i] =  \
                        (self.ts[i] / self.moving_average(i, year_period_length) + self.ts[i + year_period_length] / self.moving_average(i + year_period_length, year_period_length)) / 2
            else:
                level = level_initial
                trend = trend_initial
                day_seasonality[i] = \
                    (self.ts[i] / self.moving_average(i, day_period_length) + self.ts[i + day_period_length] / self.moving_average(i + day_period_length, day_period_length)) / 2
                week_seasonality[i] = \
                    (self.ts[i] / self.moving_average(i, week_period_length) + self.ts[i + week_period_length] / self.moving_average(i + week_period_length, week_period_length)) / 2
                if math.isnan(self.year_seasonality[i]):
                    self.year_seasonality[i] =  \
                        (self.ts[i] / self.moving_average(i, year_period_length) + self.ts[i + year_period_length] / self.moving_average(i + year_period_length, year_period_length)) / 2
     
            # Smooth 
            if i > 3 * year_period_length:
                M_0 = day_seasonality[i - day_period_length] * week_seasonality[i - week_period_length] * self.year_seasonality[i - year_period_length]
                forecast[i] = (level + trend) * M_0
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
        
        if not self.is_training: 
            # Predict
            forecast = np.full(self.horizon, math.nan)
            for i in range(len(self.ts), len(self.ts) + self.horizon):
                M_0 = day_seasonality[i - day_period_length] * week_seasonality[i - week_period_length] * self.year_seasonality[i - year_period_length]
                trend = trend_eq(beta, level, level_prev, trend_prev)
                if i == len(self.ts):
                    level = level_eq(alpha, self.ts[-1], M_0, level_prev, trend_prev)
                else:
                    level = level_eq(alpha, forecast[i - len(self.ts) - 1], M_0, level_prev, trend_prev)
                forecast[i - len(self.ts)] = (level + trend) * M_0
                day_seasonality[i] = seasonality_eq(gamma_day, forecast[i - len(self.ts)], day_seasonality[i - day_period_length], level, M_0)
                week_seasonality[i] = seasonality_eq(gamma_week, forecast[i - len(self.ts)], week_seasonality[i - week_period_length], level, M_0)
                self.year_seasonality[i] = seasonality_eq(gamma_year, forecast[i - len(self.ts)], self.year_seasonality[i - year_period_length], level, M_0)
            return forecast
        else:
            #with open('variables/year_array.npy', 'wb') as f:
            #    np.save(f, self.year_seasonality, allow_pickle=True)
            if return_error:
                return self.sse(forecast)
            else:
                return forecast

    def holt_winters_multiplicative_year_extended_predict(self, do_training):
        alpha = 0.9             # 0 <= alpha <= 1
        beta = 0.0001           # 0 <= beta <= 1
        gamma_day = 0.9         # 0 <= gamma_day <= 1
        gamma_week = 0.7        # 0 <= gamma_week <= 1
        gamma_year = 0.7        # 0 <= gamma_year <= 1

        initial_guess = np.array([alpha, beta, gamma_day, gamma_week, gamma_year])
        bound_on_variables = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))
        tolerance = 0.3
        variables_optimized = initial_guess
        self.is_training = True
        if do_training:
            variables_optimized = minimize(self.holt_winters_multiplicative_year_extended, initial_guess, bounds=bound_on_variables, tol=tolerance).x
        residuals = np.array([])
        smoothed_series = self.holt_winters_multiplicative_year_extended(variables_optimized, False)
        for i in range(0, len(self.ts)):
            residuals = np.append(residuals, self.ts[i][0] - smoothed_series[i])

        self.is_training = False
        return self.holt_winters_multiplicative_year_extended(variables_optimized), residuals


    def holt_winters_multiplicative(self, x, return_error=True):
        alpha = x[0]    # 0 <= alpha <= 1
        beta = x[1]     # 0 <= beta <= 1
        gamma = x[2]    # 0 <= gamma <= 1
        day_period_length = 24
        level_prev = self.ts[0]
        trend_prev = 1

        seasonal_component = np.full(len(self.ts) + self.horizon, math.nan)
        if self.is_training:
            forecast = np.full(len(self.ts), math.nan)
        else:
            forecast = np.full(len(self.ts) + self.horizon, math.nan)

        level_eq = lambda alpha, observation, level_prev, trend_prev, season_prev : alpha * (observation - season_prev) + (1 - alpha) * (level_prev + trend_prev)
        trend_eq = lambda beta, level, level_prev, trend_prev : beta * (level - level_prev) + (1 - beta) * trend_prev
        season_eq = lambda gamma, observation, level_prev, trend_prev, season_prev : gamma * observation / (level_prev - trend_prev) + (1 - gamma) * season_prev

        for i in range(0, len(self.ts)):
            # Calculate component
            if i - day_period_length >= 0:
                level = level_eq(alpha, self.ts[i], level_prev, trend_prev, seasonal_component[i - day_period_length])
                seasonal_component[i] = season_eq(gamma, self.ts[i], level_prev, trend_prev, seasonal_component[i - day_period_length])
            else:
                seasonal_component[i] = 1
                level = level_eq(alpha, self.ts[i], level_prev, trend_prev, 0)
            trend = trend_eq(beta, level, level_prev, trend_prev)

            # Make forecasts
            if i > 3 * day_period_length:
                forecast[i] = (level + trend) * seasonal_component[i - day_period_length]
            else:
                forecast[i] = level + trend
            level_prev = level
            trend_prev = trend

        if not self.is_training:
            # Predict
            forecast = np.full(self.horizon, math.nan)
            for i in range(len(self.ts), len(self.ts) + self.horizon):
                trend = trend_eq(beta, level, level_prev, trend_prev)
                if i == len(self.ts):
                    level = level_eq(alpha, self.ts[-1], level_prev, trend_prev, seasonal_component[i - day_period_length])
                    forecast[i - len(self.ts)] = (level + trend) * seasonal_component[i - day_period_length]
                    seasonal_component[i] = season_eq(gamma, self.ts[-1], level_prev, trend_prev, seasonal_component[i - day_period_length])
                else:
                    level = level_eq(alpha, forecast[i - len(self.ts) - 1], level_prev, trend_prev, seasonal_component[i - day_period_length])
                    forecast[i - len(self.ts)] = (level + trend) * seasonal_component[i - day_period_length]
                    seasonal_component[i] = season_eq(gamma, forecast[i - len(self.ts)], level_prev, trend_prev, seasonal_component[i - day_period_length])
                level_prev = level
                trend_prev = trend
            return forecast
        else:
            if return_error:
                return self.sse(forecast)
            else:
                return forecast
    
    def holt_winters_multiplicative_predict(self, do_training):
        alpha = 0.9     # 0 <= alpha <= 1
        beta = 0.1      # 0 <= beta <= 1
        gamma = 0.05     # 0 <= gamma <= 1

        initial_guess = np.array([alpha, beta, gamma])
        bound_on_variables = ((0, 1), (0, 1), (0, 1))
        tolerance = 0.8
        variables_optimized = initial_guess

        self.is_training = True
        if do_training:
            variables_optimized = minimize(self.holt_winters_multiplicative, initial_guess, bounds=bound_on_variables, tol=tolerance).x
        residuals = np.array([])
        smoothed_series = self.holt_winters_multiplicative(variables_optimized, False)
        for i in range(0, len(self.ts)):
            residuals = np.append(residuals, self.ts[i][0] - smoothed_series[i])

        self.is_training = False
        return self.holt_winters_multiplicative(variables_optimized), residuals

    def holt_winters_additive(self, x, return_error=True):
        alpha = x[0]    # 0 <= alpha <= 1
        beta = x[1]     # 0 <= beta <= 1
        gamma = x[2]    # 0 <= gamma <= 1
        day_period_length = 24
        level_prev = self.ts[0]
        trend_prev = 1

        seasonal_component = np.full(len(self.ts) + self.horizon, math.nan)
        if self.is_training:
            forecast = np.full(len(self.ts), math.nan)
        else:
            forecast = np.full(len(self.ts) + self.horizon, math.nan)

        level_eq = lambda alpha, observation, level_prev, trend_prev, season_prev : alpha * (observation - season_prev) + (1 - alpha) * (level_prev + trend_prev)
        trend_eq = lambda beta, level, level_prev, trend_prev : beta * (level - level_prev) + (1 - beta) * trend_prev
        season_eq = lambda gamma, observation, level_prev, trend_prev, season_prev : gamma * (observation - level_prev - trend_prev) + (1 - gamma) * season_prev

        for i in range(0, len(self.ts)):
            # Calculate component
            if i - day_period_length >= 0:
                level = level_eq(alpha, self.ts[i], level_prev, trend_prev, seasonal_component[i - day_period_length])
                seasonal_component[i] = season_eq(gamma, self.ts[i], level_prev, trend_prev, seasonal_component[i - day_period_length])
            else:
                seasonal_component[i] = 1
                level = level_eq(alpha, self.ts[i], level_prev, trend_prev, 0)
            trend = trend_eq(beta, level, level_prev, trend_prev)

            # Make forecasts
            if (i - day_period_length >= 0):
                forecast[i] = level + trend + seasonal_component[i - day_period_length]
            else:
                forecast[i] = level + trend
            level_prev = level
            trend_prev = trend

        if not self.is_training:
            # Predict
            forecast = np.full(self.horizon, math.nan)
            for i in range(len(self.ts), len(self.ts) + self.horizon):
                trend = trend_eq(beta, level, level_prev, trend_prev)
                if i == len(self.ts):
                    level = level_eq(alpha, self.ts[-1], level_prev, trend_prev, seasonal_component[i - day_period_length])
                    forecast[i - len(self.ts)] = level + trend + seasonal_component[i - day_period_length]
                    seasonal_component[i] = season_eq(gamma, self.ts[-1], level_prev, trend_prev, seasonal_component[i - day_period_length])
                else:
                    level = level_eq(alpha, forecast[i - len(self.ts) - 1], level_prev, trend_prev, seasonal_component[i - day_period_length])
                    forecast[i - len(self.ts)] = level + trend + seasonal_component[i - day_period_length]
                    seasonal_component[i] = season_eq(gamma, forecast[i - len(self.ts)], level_prev, trend_prev, seasonal_component[i - day_period_length])
                level_prev = level
                trend_prev = trend
            return forecast
        else:
            if return_error:
                return self.sse(forecast)
            else:
                return forecast
    
    def holt_winters_additive_predict(self, do_training):
        alpha = 0.9     # 0 <= alpha <= 1
        beta = 0.1      # 0 <= beta <= 1
        gamma = 0.01     # 0 <= gamma <= 1 - alpha

        initial_guess = np.array([alpha, beta, gamma])
        bound_on_variables = ((0, 1), (0, 1), (0, 1 - alpha))
        tolerance = 0.5
        
        variables_optimized = initial_guess
        self.is_training = True
        if do_training:
            variables_optimized = minimize(self.holt_winters_additive, initial_guess, bounds=bound_on_variables, tol=tolerance).x
        residuals = np.array([])
        smoothed_series = self.holt_winters_additive(variables_optimized, False)
        for i in range(0, len(self.ts)):
            residuals = np.append(residuals, self.ts[i][0] - smoothed_series[i])

        self.is_training = False
        return self.holt_winters_additive(variables_optimized), residuals
