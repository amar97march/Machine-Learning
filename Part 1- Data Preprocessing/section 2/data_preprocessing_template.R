#Importing the dataset
dataset = read.csv('Data.csv')
#missing dataset


#encoding categorical data


#splitting dataset
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) #for training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[, 2:3] = scale(test_set[,2:3])