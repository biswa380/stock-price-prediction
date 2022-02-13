# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 23:11:46 2021

@author: AMD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset
dataset = pd.read_csv("BSE-BOM500440.csv")
dataset_train = dataset.iloc[:, 4:5]
dataset_train = dataset_train.iloc[::-1]

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
dataset_train_scaled = sc.fit_transform(dataset_train)

#preparing the dataset with 20 time steps
x_train = []
y_train = []
for i in range(60, 1318):
    x_train.append(dataset_train_scaled[i-60:i, 0])
    y_train.append(dataset_train_scaled[i, 0])
x_train,y_train = np.array(x_train), np.array(y_train)

#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#importing the keras libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

#Initiallizing RNN
regressor = Sequential()

#Adding the first LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding the second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding the fourth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))

#compiling the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs=100, batch_size=32)

#Importing test data
dataset_test = pd.read_csv("test-hindalco-may-2021.csv")
real_stock_price = dataset_test.iloc[:, 4:5]
real_stock_price = real_stock_price.iloc[::-1]

#Getting the predicted stock price
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']),axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60, 83):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)

#Reshaping
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visuallizing the result
plt.plot(real_stock_price, color = 'green', label = 'real price')
plt.plot(predicted_stock_price, color = 'blue', label = 'predicted price')
plt.title('HINDALCO')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

