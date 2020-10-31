import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

#read the data file. the date column is expected to be in the mm-dd-yyyy format.
df = pd.read_csv('data/Demand_for_California_hourly_UTC_time.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
df = df.reindex(index=df.index[::-1])
df.index.freq = 'H'

#split between the training and the test data sets. The last 12 periods form the test data
test_size = 10
df_train = df.iloc[:-test_size]
df_test = df.iloc[-test_size:]

#build and train the model on the training data
model = HWES(df_train, seasonal_periods=24, trend='add', seasonal='mul')
fitted = model.fit(optimized=True, use_brute=True)

#print out the training summary
print(fitted.summary())

#create an out of sample forcast for the next $test_size steps beyond the final data point in the training data set
sales_forecast = fitted.forecast(steps=test_size)

#plot the training data, the test data and the forecast on the same plot
fig = plt.figure()
fig.suptitle('Retail Sales of Used Cars in the US (1992-2020)')
past, = plt.plot(df_train.index, df_train, 'b.-', label='Consumption History')
future, = plt.plot(df_test.index, df_test, 'r.-', label='Actual Consumption')
predicted_future, = plt.plot(df_test.index, sales_forecast, 'g.-', label='California\'s energy consumption')
plt.legend(handles=[past, future, predicted_future])
plt.show()
