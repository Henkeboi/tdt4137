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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from scipy.optimize import minimize

class Hybrid:
    def __init__(self, lstm, ts, horizon):
        self.ts = ts
        self.horizon = horizon
        self.is_training = True
        self.year_seasonality = np.full(len(self.ts) + self.horizon, math.nan)
        #self.year_seasonality = np.load('variables/year_array.npy')
        while len(self.year_seasonality) < len(ts) + horizon:
            self.year_seasonality = np.append(self.year_seasonality, math.nan)
        self.lstm = lstm

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
        level_prev = self.ts[0]

        seasonal_component = np.full(len(self.ts) + self.horizon, math.nan)
        if self.is_training:
            forecast = np.full(len(self.ts), math.nan)
        else:
            forecast = np.full(len(self.ts) + self.horizon, math.nan)

        level_eq = lambda alpha, observation, level_prev, season_prev : alpha * (observation / season_prev) + (1 - alpha) * level_prev 
        season_eq = lambda gamma, observation, level_prev, season_prev : gamma * observation / level_prev + (1 - gamma) * season_prev

        for i in range(0, len(self.ts)):
            # Calculate component
            if i - 3 * day_period_length >= 0:
                level = level_eq(alpha, self.ts[i], level_prev, seasonal_component[i - day_period_length])
                seasonal_component[i] = season_eq(gamma, self.ts[i], level_prev, seasonal_component[i - day_period_length])
            else:
                seasonal_component[i] = (self.ts[i] / self.moving_average(i, 12) + self.ts[i + 12] / self.moving_average(i + 12, 12)) / 2
                level = level_eq(alpha, self.ts[i], level_prev, 1)

            # Make forecasts
            if i > 3 * day_period_length:
                forecast[i] = level * seasonal_component[i - day_period_length]
            else:
                forecast[i] = level 
            level_prev = level

        if not self.is_training:
            # Predict
            forecast = np.full(self.horizon, math.nan)
            for i in range(len(self.ts), len(self.ts) + self.horizon):
                if i == len(self.ts):
                    level = level_eq(alpha, self.ts[-1], level_prev, seasonal_component[i - day_period_length])
                    forecast[i - len(self.ts)] = level * seasonal_component[i - day_period_length]
                    seasonal_component[i] = season_eq(gamma, self.ts[-1], level_prev, seasonal_component[i - day_period_length])
                else:
                    level = level_eq(alpha, forecast[i - len(self.ts) - 1], level_prev, seasonal_component[i - day_period_length])
                    forecast[i - len(self.ts)] = level * seasonal_component[i - day_period_length]
                    seasonal_component[i] = season_eq(gamma, forecast[i - len(self.ts)], level_prev, seasonal_component[i - day_period_length])
                level_prev = level
            return forecast
        else:
            if return_error:
                return self.sse(forecast)
            else:
                return forecast
    
    def holt_winters_multiplicative_predict(self, do_training):
        alpha = 0.9     # 0 <= alpha <= 1
        gamma = 0.9      # 0 <= beta <= 1

        initial_guess = np.array([alpha, gamma])
        bound_on_variables = ((0, 1), (0, 1))
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


def get_training_vectors(training_data, window_size):
    #Splitting the data
    x_training = []
    y_training = []

    for i in range(window_size, len(training_data)):
        x_training.append(training_data[i - window_size : i, 0])
        y_training.append(training_data[i, 0])

    #Convert to numpy arrays
    x_training, y_training = np.array(x_training), np.array(y_training)

    #Reshape the data into 3-D array
    x_training = np.reshape(x_training, (x_training.shape[0], x_training.shape[1], 1))
    return x_training, y_training

def build_and_train_model(features, window_size):
    training_dataset_length = math.ceil(len(features) * .75)
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(features)
    training_data = scaled_data[0 : training_dataset_length, :]

    x_training, y_training = get_training_vectors(training_data, window_size)

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
    model.fit(x_training, y_training, epochs=5, batch_size=800)

    return model

def get_test_data(scaled_data, training_dataset_length, features, window_size):
    #Test data set
    test_data = scaled_data[training_dataset_length - window_size:, : ]

    #splitting the x_test and y_test data sets
    x_test = []
    y_test = features[training_dataset_length :, : ] 

    for i in range(window_size, len(test_data)):
        x_test.append(test_data[i - window_size : i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test

def test_model(model, features, window_size):
    training_dataset_length = math.ceil(len(features) * .75)
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(features)

    x_test, y_test = get_test_data(scaled_data, training_dataset_length, features, window_size)

    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions.flatten()
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(rmse)
    x = np.linspace(0, 100 * len(predictions), len(predictions))
    sns.set(rc={'figure.figsize':(16, 4)})

    sns.lineplot(data=predictions)
    sns.lineplot(data=y_test)
    plt.show()

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    horizon = 24
    training_set = data.loc['2015-07-1' : '2015-10-31'].to_numpy()
    window_size = 10
    #model = build_and_train_model(training_set, window_size)

    validation_set = data.loc['2015-11-1' : '2015-11-30'].to_numpy()
    #test_model(model, validation_set, window_size)
    model = Sequential()
    hybrid = Hybrid(model, training_set, len(validation_set))
    prediction, residuals = hybrid.holt_winters_multiplicative_predict(False)
    sns.set(rc={'figure.figsize':(16, 4)})
    sns.lineplot(data=prediction)
    plt.show()

if __name__ == '__main__':
    main()
