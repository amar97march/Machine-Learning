# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:39:55 2018

@author: amar97march
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2:3].values

# data is very less that is why only training
#polynomial automnatically uses feature scaling

#firstly using the linear regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() 
lin_reg.fit(x,y)

#fitting Polymial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

ply_reg = PolynomialFeatures(degree = 4)
x_poly = ply_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#Visualizing the linear
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Label')
plt.ylabel('Salaries')
plt.show()


#Visualizing the Plynomial regression
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid)),1)

plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg2.predict(ply_reg.fit_transform(x_grid)),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Label')
plt.ylabel('Salaries')
plt.show()

#Predicting
lin_reg.predict(6.5)
lin_reg2.predict(ply_reg.fit_transform(6.5))