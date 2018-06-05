# ANN--Kaggle-Competition
ANN built to predict passenger survival on the Titanic

# Kaggle Titanic Survivors
# Binary Clasification ANN

# Importing the dataset
titanic_dataset = read.csv('train.csv')
titanic_dataset = titanic_dataset[1:7]

# Encoding the Categorical Variables as factor

titanic_dataset$Sex = as.numeric(factor(titanic_dataset$Sex,
                                      levels = c('male','female'),
                                      labels = c(1,2)))

# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)

set.seed(123)
split = sample.split(titanic_dataset$Survived, SplitRatio = 0.8)
training_set = subset(titanic_dataset, split == TRUE)
test_set = subset(titanic_dataset, split == FALSE)

# Feature Scaling
training_set[-7] = scale(training_set[-7])
test_set[-7] = scale(test_set[-7])


# Fitting ANN to the Training set
#install.packages('h2o')
#library(h2o)
h2o.init(nthreads = -1)

#Fitting to ANN
classifier = h2o.deeplearning(y = 'Survived',
                              training_frame = as.h2o(titanic_dataset),
                              activation = 'Rectifier',
                              hidden = c(4,4),
                              epochs = 100,
                              train_samples_per_iteration = -2)




# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-7]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)
y_pred

# Making the Confusion Matrix
cm = table(test_set[, 7], y_pred)
cm

# Accuracy

(95+45)/178

h2o.shutdown()

# >>>>>>>>>>>>> MAKING PREDICTIONS  <<<<<<<<

# Titanic ANN Prediction

# Importing the datasets
titanic_dataset = read.csv('train.csv')
titanic_dataset = titanic_dataset[1:7]

titanic_test = read.csv('test_full.csv')
titanic_test = titanic_test[1:7]

# Encoding the Categorical Variables as factor

titanic_dataset$Sex = as.numeric(factor(titanic_dataset$Sex,
                                        levels = c('male','female'),
                                        labels = c(1,2)))



# Feature Scaling
titanic_dataset[-7] = scale(titanic_dataset[-7])


# >>>>>>>>>>>>>>>>>>>>>>>>>>>> FITTING ANN - TRAINING <<<<<<<<<<<<<<<<<<<<<<<<<

#install.packages('h2o')
#library(h2o)
h2o.init(nthreads = -1)

#Fitting to ANN
classifier = h2o.deeplearning(y = 'Survived',
                              training_frame = as.h2o(titanic_dataset),
                              activation = 'Rectifier',
                              hidden = c(4,4),
                              epochs = 100,
                              train_samples_per_iteration = -2)

# Preprocessing for Test

# Encoding Catagorical as Numeric for Test

titanic_test$Sex = as.numeric(factor(titanic_test$Sex,
                                     levels = c('male','female'),
                                     labels = c(1,2)))

# Feature Scaling for test

titanic_test[-7] = scale(titanic_test[-7])

# Predicting Results

# Predicting the Test set results
prob_pred = h2o.predict(classifier, newdata = as.h2o(titanic_test[-7]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)
y_pred

test_full = read.csv('test_full.csv')
final <- as.data.frame(y_pred, row.names = NULL)
Answer = cbind(final,test_full)
Answer

write.csv(Answer,'Titanic_Submission.csv')


# https://www.kaggle.com/c/titanic


