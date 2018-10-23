# -*- coding: utf-8 -*-
"""
Created on Wed May  9 07:26:52 2018

@author: amar97march
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset = pd.read_excel("Data.xlsx")
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
X_test = X_train.reshape(-1,1)
Y_test = Y_train.reshape(-1,1)

#regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


y_pred = regressor.predict(X_test)

#ploting graph
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Hours Practiced v\s Marks Obtained')
plt.xlabel('Hours Practiced')
plt.ylabel('marks obtained')
plt.show()