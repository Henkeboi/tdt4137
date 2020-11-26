import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GaussianNoise, Activation
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize


class Hybrid:
    def __init__(self, ts, horizon):
        self.ts = ts
        self.horizon = horizon
        self.smoothed_ts = np.full(len(self.ts), math.nan)
        self.lstm = None
        self.seasonal_component = np.full(len(self.ts) + self.horizon, math.nan)
        self.level_prev = self.ts[0]

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

    def holt_winters_multiplicative(self, x, return_error=True):
        alpha = x[0]    # 0 <= alpha <= 1
        gamma = x[1]    # 0 <= gamma <= 1
        day_period_length = 24

        forecast = np.full(len(self.ts), 1)

        level_eq = lambda alpha, observation, level_prev, season_prev : alpha * observation / season_prev + (1 - alpha) * level_prev
        season_eq = lambda gamma, observation, level_prev, season_prev : gamma * observation / level_prev + (1 - gamma) * season_prev

        for i in range(0, len(self.ts)):
            # Calculate component
            if i - 3 * day_period_length >= 0:
                level = level_eq(alpha, self.ts[i], self.level_prev, self.seasonal_component[i - day_period_length])
                self.seasonal_component[i] = season_eq(gamma, self.ts[i], self.level_prev, self.seasonal_component[i - day_period_length])
            else:
                self.seasonal_component[i] = (self.ts[i] / self.moving_average(i, 12) + self.ts[i + 12] / self.moving_average(i + 12, 12)) / 2
                level = level_eq(alpha, self.ts[i], self.level_prev, 1)

            # Make forecasts
            if i > 3 * day_period_length:
                forecast[i] = level * self.seasonal_component[i - day_period_length]
            else:
                forecast[i] = level 
            self.smoothed_ts[i] = self.ts[i] / forecast[i]
            self.level_prev = level
    
    def holt_winters_smooth(self):
        alpha = 0.2     # 0 <= alpha <= 1
        gamma = 0.2      # 0 <= beta <= 1
        self.is_training = True
        self.holt_winters_multiplicative([alpha, gamma], False)
        self.smoothed_ts = self.smoothed_ts.reshape(self.smoothed_ts.shape[0], 1) 

    def predict(self):
        alpha = 0.2     # 0 <= alpha <= 1
        gamma = 0.2      # 0 <= beta <= 1
        day_period_length = 24

        forecast = np.full(self.horizon, math.nan)
        level_eq = lambda alpha, observation, level_prev, season_prev : alpha * observation / season_prev + (1 - alpha) * level_prev
        season_eq = lambda gamma, observation, level_prev, season_prev : gamma * observation / level_prev + (1 - gamma) * season_prev

        for i in range(len(self.ts), len(self.ts) + self.horizon):
            if i == len(self.ts):
                level = level_eq(alpha, self.ts[0][-1], self.level_prev, self.seasonal_component[i - day_period_length])
                lstm_val = self.lstm_step()
                forecast[i - len(self.ts)] = level * self.seasonal_component[i - day_period_length] * lstm_val
                smoothed_val = self.ts[0][-1] / (level * self.seasonal_component[i - day_period_length])
                self.smoothed_ts = np.append(self.smoothed_ts, smoothed_val)
                self.seasonal_component[i] = season_eq(gamma, self.ts[0][-1], self.level_prev, self.seasonal_component[i - day_period_length])
            else:
                level = level_eq(alpha, forecast[i - len(self.ts) - 1], self.level_prev, self.seasonal_component[i - day_period_length])
                lstm_val = self.lstm_step()
                forecast[i - len(self.ts)] = level * self.seasonal_component[i - day_period_length] * lstm_val
                self.seasonal_component[i] = season_eq(gamma, forecast[i - len(self.ts)], self.level_prev, self.seasonal_component[i - day_period_length])
                self.smoothed_ts = np.append(self.smoothed_ts, self.ts[0][-1] / (level * self.seasonal_component[i - day_period_length]) )
            self.level_prev = level
        return forecast

    def lstm_step(self):
        x = self.smoothed_ts[-1] 
        x = x.reshape(1, 1, 1)
        prediction = self.lstm.predict(x, batch_size=1)
        return prediction.flatten()[0]

    def init_lstm(self):
        self.lstm.reset_states()
        for i in range(0, 7):
            x = self.smoothed_ts[-i] 
            x = x.reshape(1, 1, 1)
            prediction = self.lstm.predict(x, batch_size=1)

    def forecast(self):
        self.holt_winters_smooth()
        self.build_and_train_model()
        self.init_lstm()
        return self.predict()

    def get_residuals(self):
        self.lstm.reset_states()

        alpha = 0.2     # 0 <= alpha <= 1
        gamma = 0.2      # 0 <= beta <= 1
        day_period_length = 24

        residuals = np.full(len(self.ts), math.nan)
        level_eq = lambda alpha, observation, level_prev, season_prev : alpha * observation / season_prev + (1 - alpha) * level_prev
        season_eq = lambda gamma, observation, level_prev, season_prev : gamma * observation / level_prev + (1 - gamma) * season_prev

        for i in range(len(self.ts) - 24 * 7 * 52, len(self.ts)):
            print(i)
            if i == len(self.ts):
                level = level_eq(alpha, self.ts[-1], self.level_prev, self.seasonal_component[i - day_period_length])
                lstm_val = self.lstm_step()
                residuals[i] = self.ts[i] - level * self.seasonal_component[i - day_period_length] * lstm_val
                smoothed_val = self.ts[i] / (level * self.seasonal_component[i - day_period_length])
                self.smoothed_ts = np.append(self.smoothed_ts, smoothed_val)
                self.seasonal_component[i] = season_eq(gamma, self.ts[-1], self.level_prev, self.seasonal_component[i - day_period_length])
            else:
                level = level_eq(alpha, -residuals[i - 1] + self.ts[i - 1], self.level_prev, self.seasonal_component[i - day_period_length])
                lstm_val = self.lstm_step()
                residuals[i] = self.ts[i] - level * self.seasonal_component[i - day_period_length] * lstm_val
                self.seasonal_component[i] = season_eq(gamma, residuals[i], self.level_prev, self.seasonal_component[i - day_period_length])
                self.smoothed_ts = np.append(self.smoothed_ts, self.ts[-1] / (level * self.seasonal_component[i - day_period_length]) )
            self.level_prev = level

        return residuals

    def build_and_train_model(self):
        df = pd.DataFrame(self.smoothed_ts[1000:])
        df = pd.concat([df, df.shift(1)], axis=1)
        df.dropna(inplace=True) # Remove NaN
        #length = int(1680 / 2)
        length = int(24 * 7 * 52 * 2)
        x_train, y_train = df.values[0:length, 0], df.values[0:length, 1]
        x_train = x_train.reshape(len(x_train), 1, 1) # [Samples, Time steps, Features]

        batches = 24 * 7
        model = Sequential()
        model.add(LSTM(units=50, batch_input_shape=(batches, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dropout(0.4))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train
        for i in range(20):
            model.fit(x_train, y_train, epochs=1, batch_size=batches, shuffle=False, verbose=0)
            model.reset_states()

        # We need a new model with a batch size of 1
        batches = 1 
        self.lstm = Sequential()
        self.lstm.add(LSTM(units=50, batch_input_shape=(batches, x_train.shape[1], x_train.shape[2]), stateful=True))
        self.lstm.add(Dropout(0.4))
        self.lstm.add(Dense(units=1))
        self.lstm.set_weights(model.get_weights())
        self.lstm.compile(optimizer='adam', loss='mean_squared_error')
        
def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    training_set = data.loc['2019-01-1' : '2019-03-31'].to_numpy()
    validation_set = data.loc['2019-04-1' : '2019-04-7'].to_numpy()
    hybrid = Hybrid(training_set, len(validation_set))
    forecast = hybrid.forecast()
    rmse = np.sqrt(np.mean(((forecast - validation_set) ** 2)))
    print(rmse)
    
    sns.set(rc={'figure.figsize':(16, 4)})
    sns.lineplot(data=validation_set, color="b")
    sns.lineplot(data=forecast, color="r")
    plt.show()

if __name__ == '__main__':
    main()
