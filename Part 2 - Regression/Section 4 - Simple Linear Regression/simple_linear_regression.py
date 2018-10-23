#simple linear regression

import numpy as np #for math
import matplotlib.pyplot as plt # for plotting
import pandas as pd # import and manage dataset

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
# missing data


#encoding categorical data


#splitting data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set results
Y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Expirience (Training set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Expirience (Test set)')
plt.xlabel('Years of expirience')
plt.ylabel('Salary')
plt.show()
