#importing the libraies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing hte dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#Using the elbow method to find the number of clustrs
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++',max_iter= 300,n_init= 10, random_state= 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()
#applying k-means
kmeans = KMeans(n_clusters= 5, init='k-means++',max_iter= 300,n_init= 10,random_state= 0)
y_pred = kmeans.fit_predict(x)

# Visualization the cluster
plt.scatter(x[y_pred == 0,0],x[y_pred == 0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[y_pred == 1,0],x[y_pred == 1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[y_pred == 2,0],x[y_pred == 2,1],s=100,c='green',label='Cluster 3')
plt.scatter(x[y_pred == 3,0],x[y_pred == 3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(x[y_pred == 4,0],x[y_pred == 4,1],s=100,c='magenta',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300,c = 'yellow',label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()