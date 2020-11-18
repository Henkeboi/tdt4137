import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

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

def build_and_train_model(x_training, y_training):
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
    model.fit(x_training, y_training, epochs=1, batch_size=200)
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

def main():
    data = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
    data = data.reindex(index=data.index[::-1])
    data.index.freq = 'H' # Hourly data.
    horizon = 24
    features = data.loc['2015-07-1' : '2015-10-31'].to_numpy()

    training_dataset_length = math.ceil(len(features) * .75)
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(features)
    training_data = scaled_data[0 : training_dataset_length, :]

    window_size = 10
    x_training, y_training = get_training_vectors(training_data, window_size)
    model = build_and_train_model(x_training, y_training)     
    x_test, y_test = get_test_data(scaled_data, training_dataset_length, features, window_size)

    predictions = model.predict(x_test) 
    #Undo scaling
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions.flatten()
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    x = np.linspace(0, 100 * len(predictions), len(predictions))
    sns.set(rc={'figure.figsize':(16, 4)})

    sns.lineplot(data=predictions)
    sns.lineplot(data=y_test)
    plt.show()

if __name__ == '__main__':
    main()
