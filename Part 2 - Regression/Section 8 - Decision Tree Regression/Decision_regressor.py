import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importng dataset
Dataset  = pd.read_csv('Position_Salaries.csv')
x = Dataset.iloc[:,1:2].values
y = Dataset.iloc[:,2:3].values

#Fitting the Decision tree reressor to the dataset

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

#Predicting the ne values

y_pred = regressor.predict(6.5)

#potting the graph
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Truth or Bluff (Decision Tree Regressor)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()