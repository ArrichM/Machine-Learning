###################################
# Machine Learning - Homework 1   #
# Maximilian Arrich #
# Florian Benkhalifa #
###################################
rm(list=ls())
library(MASS)

### Exercise 1.1 ###
set.seed(100)
data   <- Boston
sample <- sample(nrow(data), floor(0.7*nrow(data)), replace = FALSE) #we declare 70% of the data to be training data
train  <- data[sample,][,13:14]
test   <- data[-sample,][,13:14]




### Exercise 1.2 ###
fit_train <- lm(medv~lstat, data=train)
fit_train
# The prediction function is given by: medv = 35.0073 - 0.9752*lstat




### Exercise 1.3 ###
#using predictinon formula
pred_test <- fit_train$coefficients[1] + fit_train$coefficients[2]*test$lstat

#alternatively, we can obtain the prediction by using the predicting function:
pred_test_alternative <- predict(fit_train, newdata = test)

#we get the same predicitons from both methods
all(pred_test == pred_test_alternative)




### Exercise 1.4 ###
#the residuals is simply the difference of the empirical values and our predictions:
residuals <- test$medv-pred_test


### Exercise 1.5 ###
# For the sake of handiness concerning the following Exercises we decided to condense the previous commands to one single 
# function. This allows us to compute RSS, RSE and R-squared, while diffentiating between linear and quadratic regression 
# models by choosing the poly variable accordingly. Further, specifying the seed variable allows to collect different random
# samples in Exercise 2.2 and 2.3. while attaining reproducibility. We did not fully understand if the notion of "different 
# random splittings into training and test data" in question 2.2 was related to randomly varying the test and training sizes,
# or to keep the splitting ratio (70%-30%) and to simply take 5 different random samples. Because of that, we included the
# possibility to also vary the partition sizes fixed splitting by setting the training size randomly between 0.5 and 1. 
# The function looks as follows:

fun <- function(poly, seed, varying_partition = F){
  set.seed(seed)    # Setting seed to create multiple reproducible samples
  # Create random partition size.
  if(varying_partition == T) training_size <- 0.5 + runif(1,0,0.5) else training_size <- 0.7  
  # Creating sample
  sample    <- sample(nrow(data), floor(training_size*nrow(data)), replace = FALSE)    
  train     <- data[sample,][,13:14]
  test      <- data[-sample,][,13:14]
  # Regression formula
  formula   <- paste("medv~poly(lstat,", as.character(poly), ",raw=T)")  
  lm        <- lm(as.formula(formula), data=train)
  predict   <- predict(lm, newdata = test)
  residuals <- test$medv - predict
  RSS       <- sum(residuals^2)
  RSE       <- sqrt(RSS/(nrow(data)-2))
  R_squared <- 1 - RSS/(var(test$medv)*(nrow(data)-1))
  aic       <- AIC(lm)
  # organizing output in matrix
  mat       <- matrix(c(RSS, RSE, R_squared,aic), nrow=4, 
                      dimnames = list(c("RSS", "RSE", "R_squared","AIC")))
  return(mat)
}

#obtain R^2 for model from Exercise 1:
fun(poly=1, seed=100)[3,1]



###### Exercise 2 ######

### Ecercise 2.1 ###
# Computing quadratic regression by specifying poly=2
fun(poly=2, seed= 100)




# ##Exercise 2.2
# Here, we interpreted "different random splittings into training and test data" as keeping the partition size fixed while simply taking 5 different random samples. Below, we also added an alternative solution where we allowed the partition size to be randomly selected.
# We compute the desired statistic for five different random data splittings:
# Linear Model
R2_linear <- do.call(cbind,lapply(1:5, fun, poly = 1)) #values for linear model

# Quadratic Model
R2_quadratic <- do.call(cbind,lapply(1:5, fun, poly = 2)) #values for quadratic model




### Ecercise 2.3 ###
mean(R2_linear[3,])
mean(R2_quadratic[3,])

# Since, on average, the quadratic model explains more of the variance on the testing data (higher R^2), it seems eligible to choose the quadratic model. 
# This choice is supported by the average AIC of the quadratic model being lower than the one of the linear model, implying a lower Kullback-Leiber divergence:

mean(R2_linear[4,])
mean(R2_quadratic[4,])

#Alternative to Exercise 2.2 with random splitting ratios
# Now, we present an alternative way of understanding "different random splittings" by also varying the size of the partitions. 
# However, even by using this alternative interpretation, the quadratic model prevails as we will see below.


### Ecercise 2.2 ###
# Linear Model
R2_linear_alt <- do.call(cbind,lapply(1:5, fun, poly = 1, varying_partition = 1)) #values for linear model

# Quadratic Model
R2_quadratic_alt <- do.call(cbind,lapply(1:5, fun, poly = 2, varying_partition = 1)) #values for quadratic model


### Ecercise 2.3 ###
mean(R2_linear_alt[3,])
mean(R2_quadratic_alt[3,])

# Since, on average, the quadratic model explains more of the variance on the testing data (higher R^2), it seems eligible to choose the quadratic model. 
# This choice is supportet by the AIC of the quadratic model being lower than the one of the linear model, implying a lower Kullback-Leiber divergence:

mean(R2_linear_alt[4,])
mean(R2_quadratic_alt[4,])




