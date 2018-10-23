#Simple linear regression
#Importing the dataset
dataset = read.csv('Salary_Data.csv')
#missing dataset


#encoding categorical data


#splitting dataset
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3) #for training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[, 2:3] = scale(test_set[,2:3])

#Fitting simple linear regressio to training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#Predicting the test set results
Y_pred = predict(regressor, newdata = test_set)

#Visualing the training set results
# install.packages('ggplot2')

ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs YearsExperience (training set)') + 
  xlab('YearsExperience') + 
  ylab('salary')

#Visualing the test set results
# install.packages('ggplot2')

ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs YearsExperience (training set)') + 
  xlab('YearsExperience') + 
  ylab('salary')