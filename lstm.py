import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
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
        self.window_size = 20 # Sliding winding for the lstm
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
        alpha = 0.9     # 0 <= alpha <= 1
        gamma = 0.9      # 0 <= beta <= 1
        self.is_training = True
        self.holt_winters_multiplicative([alpha, gamma], False)
        self.smoothed_ts.reshape(self.smoothed_ts.shape[0], 1) 


    def predict(self):
        alpha = 0.9     # 0 <= alpha <= 1
        gamma = 0.9      # 0 <= beta <= 1
        day_period_length = 24

        forecast = np.full(self.horizon, math.nan)
        level_eq = lambda alpha, observation, level_prev, season_prev : alpha * observation / season_prev + (1 - alpha) * level_prev
        season_eq = lambda gamma, observation, level_prev, season_prev : gamma * observation / level_prev + (1 - gamma) * season_prev

        for i in range(len(self.ts), len(self.ts) + self.horizon):
            if i == len(self.ts):
                level = level_eq(alpha, self.ts[0][-1], self.level_prev, self.seasonal_component[i - day_period_length])
                lstm_val = self.lstm_step()
                forecast[i - len(self.ts)] = level * self.seasonal_component[i - day_period_length] * lstm_val
                smoothed_val = self.ts[0][-1] / forecast[i - len(self.ts)]
                self.smoothed_ts = np.append(self.smoothed_ts, smoothed_val)
                self.seasonal_component[i] = season_eq(gamma, self.ts[0][-1], self.level_prev, self.seasonal_component[i - day_period_length])
            else:
                level = level_eq(alpha, forecast[i - len(self.ts) - 1], self.level_prev, self.seasonal_component[i - day_period_length])
                lstm_val = self.lstm_step()
                forecast[i - len(self.ts)] = level * self.seasonal_component[i - day_period_length] * lstm_val
                self.seasonal_component[i] = season_eq(gamma, forecast[i - len(self.ts)], self.level_prev, self.seasonal_component[i - day_period_length])
                self.smoothed_ts = np.append(self.smoothed_ts, lstm_val)
            self.level_prev = level
        return forecast

    def lstm_step(self):
        lstm_data = self.smoothed_ts[-100:]
        lstm_data = lstm_data.reshape(100, 1)

        training_dataset_length = len(lstm_data) - 1
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(lstm_data)
         
        lstm_data, _ = self.get_test_vectors(scaled_data, training_dataset_length, lstm_data, self.window_size)
        prediction = self.lstm.predict(lstm_data) 
        prediction = scaler.inverse_transform(prediction)
        prediction = prediction.flatten()
        return prediction

    def forecast(self):
        self.holt_winters_smooth()
        self.build_and_train_model()
        return self.predict()

    def build_and_train_model(self):
        features = self.ts
        training_dataset_length = math.ceil(len(features) * .75)
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        scaled_data = scaler.fit_transform(features)
        training_data = scaled_data[0 : training_dataset_length, :]

        x_training, y_training = self.get_training_vectors(training_data, self.window_size)

        # Initialising the RNN
        model = Sequential()
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_training.shape[1], 1)))
        model.add(Dropout(0.2))

        # Adding a second LSTM layer and Dropout layer
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        # Adding a third LSTM layer and Dropout layer
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        # Adding a fourth LSTM layer and and Dropout layer
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_training, y_training, epochs=10, batch_size=200)
        self.lstm = model

    def get_training_vectors(self, training_data, window_size):
        x_training = []
        y_training = []

        for i in range(window_size, len(training_data)):
            x_training.append(training_data[i - window_size : i, 0])
            y_training.append(training_data[i, 0])

        x_training, y_training = np.array(x_training), np.array(y_training)

        x_training = np.reshape(x_training, (x_training.shape[0], x_training.shape[1], 1))
        return x_training, y_training

    def get_test_vectors(self, scaled_data, training_dataset_length, features, window_size):
        test_data = scaled_data[training_dataset_length - window_size:, :]

        x_test = []
        y_test = features[training_dataset_length :, : ] 

        for i in range(window_size, len(test_data)):
            x_test.append(test_data[i - window_size : i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    horizon = 24
    training_set = data.loc['2016-01-1' : '2016-03-31'].to_numpy()
    validation_set = data.loc['2016-04-1' : '2016-04-10'].to_numpy()
    hybrid = Hybrid(training_set, len(validation_set))
    forecast = hybrid.forecast()
    
    sns.set(rc={'figure.figsize':(16, 4)})
    sns.lineplot(data=validation_set, color="b")
    sns.lineplot(data=forecast, color="r")
    plt.show()

if __name__ == '__main__':
    main()
