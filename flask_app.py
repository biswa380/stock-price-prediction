# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 00:37:17 2022

@author: AMD
"""

import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras
import numpy as np

app = Flask(__name__)
model = keras.models.load_model("my_model.h5")

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
@app.route("/predict", methods=["GET"])
def predict():
    dataset = pd.read_csv("HINDALCO_TRAIN_SET.csv")
    dataset_train = dataset.iloc[:, 8:9]
    dataset_train_scaled = sc.fit_transform(dataset_train)
    dataset_test = pd.read_csv("HINDALCO_TEST_SET.csv")
    real_stock_price = dataset_test.iloc[:, 8:9]
    dataset_total = pd.concat((dataset_train['Close Price'], dataset_test['Close Price']),axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    
    x_test=[]
    for i in range(60, 87):
        x_test.append(inputs[i-60:i, 0])
    x_test = np.array(x_test)

    #Reshaping
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
              
    prediction=model.predict(x_test)
    prediction = sc.inverse_transform(prediction)
    return jsonify({"predicted_price": str(prediction)})


if __name__ == "__main__":
    app.run()




