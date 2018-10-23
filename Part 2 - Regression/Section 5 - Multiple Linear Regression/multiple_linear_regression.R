#Importing the dataset
dataset = read.csv('50_Startups.csv')
#missing dataset


#encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))


#splitting dataset
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8) #for training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[, 2:3] = scale(test_set[,2:3])

#Fitting multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

# regressor = lm(formula = Profit ~ R.D.Spend,
#                data = training_set)

#Predicting the test set results

Y_pred = predict(regressor, newdata = test_set)

#Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)
