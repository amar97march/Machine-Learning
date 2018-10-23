#Upper Confidence bound

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implemeting the UCB
import math
N = 10000
d = 10
ad_selected = []
number_of_selection = [0] * d
sum_of_reward = [0] * d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (number_of_selection[i] > 0):
            average_reward = sum_of_reward[i]/number_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n) / number_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ad_selected.append(ad)
    number_of_selection[ad] = number_of_selection[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_reward[ad] = sum_of_reward[ad] + reward
    total_reward = total_reward + reward
    
#Visualizing the results
plt.hist(ad_selected,color = 'black')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad is selected')
plt.title('Histogram of ads selections')
plt.show()