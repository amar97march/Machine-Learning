import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importng dataset
Dataset  = pd.read_csv('Position_Salaries.csv')
x = Dataset.iloc[:,1:2].values
y = Dataset.iloc[:,2:3].values

#Fitting the RandomForest Regressor to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state= 0)
regressor.fit(x,y)

#Predicting the model
y_prid = regressor.predict(6.5)

#Plotting the graphs
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))


plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.title('Position Level v/s Salaries')
plt.show()

