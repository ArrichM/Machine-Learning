# *************************************************************** #
# Machine Learning              : Group Project                   #
# Code File                     : Main                            #
# Student IDs                   : 15605017  - 13614011            #
# *************************************************************** #





# ============================== Library Calls  ==============================

toload <- c("magrittr","plyr","reshape2","neuralnet","randomForest","glmnet")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
sapply(toinstall, install.packages, character.only = TRUE)
sapply(toload, require, character.only = TRUE)





# ============================== Read Data  ==============================


## Working Directory Setting
wd <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(wd)

## Read Data from CSV
dat <- read.csv("/Users/Max/Dropbox/Risikomanagement/Uni/ML/LGD/mortgage.csv") %T>% attach

## Remove NA observations
dat <- dat[complete.cases(dat),]

dat$age <- dat$time- dat$orig_time








# ============================== Prepare Data ==============================


# Scale data for neural network, we do not scale id and time
max <-  apply(dat[,-c(1,2)], 2 , max)
min <-  apply(dat[,-c(1,2)], 2 , min)
scaled_data <-  cbind(dat[,c(1,2)], as.data.frame(scale(dat[,-c(1,2)], center = min, scale = max - min)))


# We select a subset of the data and split it into training and testing
set.seed(100)

# Sample n data at random
temp_data <- scaled_data[sample(1:nrow(scaled_data),5000),]


# Select 2/3 of the data as trainign data
index <- sample(1:nrow(temp_data), (nrow(temp_data)*0.6) %>% ceiling)

train_data <- temp_data[index,][,-c(1:4,22,23)]

test_data <- temp_data[-index, ][,-c(1:4,22,23)]










# ============================== Generate Neural Network  ==============================

fit_nn <- function(layers = c(1,1), train_data1 = train_data, test_data1 = test_data, tr = 0.02, create_network = F){
  
  # Fit neural network
  if(create_network == F){
  #fit network
  nn <- neuralnet(default_time ~ ., data = train_data1, hidden = layers, act.fct = "logistic", linear.output = F,
          err.fct = "sse", lifesign = "full", threshold = tr, algorithm = "sag", learningrate.factor = list( minus = 0.5, plus = 1.2))
  }else{
  #fit network and save to global environment
  nn <<-  neuralnet(default_time ~ ., data = train_data1, hidden = layers, act.fct = "logistic", linear.output = F,
              err.fct = "sse", lifesign = "full", threshold = tr, algorithm = "sag", learningrate.factor = list( minus = 0.5, plus = 1.2))
    
  }

  # Compute MSE from probabilities
  mean((predict(nn, test_data1) - test_data$default_time) ^2) %T>% return

}

# Evaluate a number of possible layer / neuron combinations
eval <- apply(expand.grid(4:10,3:6),1, fit_nn, tr = 0.02)







# Fit best network and save under name "nn"
fit_nn(expand.grid(4:10,3:6)[which.min(eval),] %>% unlist, create_network = T)

# Get predictions for the testing set
predicted_nn <- predict(nn, test_data1) 
predicted_nn_bin <- ifelse(predicted_nn > 0.5, 1,0)

# How many defualts were predicted?
table(predicted_nn_bin)

# Compare with actual data, did we get something right?
compare <- cbind(predicted_nn_bin, test_data1$default_time)[which(test_data1$default_time == 1),] %T>% print

# How many did we get wrong / right
table(predicted_nn_bin == test_data$default_time)



# ============================== Logistic Regression  ==============================


#We perform logistic regression as benchmark model
log_reg <- glm(default_time ~ ., data = train_data, family = "binomial")

# Get predictions for the testing set
predicted_reg <- predict(log_reg, newdata = test_data)
predicted_reg_bin <- ifelse(predicted_reg > 0.5,1,0)

# How many defualts were predicted?
table(predicted_reg_bin) 

# Compare with actual data, did we get something right?
compare_reg <- cbind(predicted_reg_bin, test_data$default_time)[which(test_data$default_time == 1),] %T>% print

# Compute MSE from probabilities
mse_log <- mean((predicted_reg - test_data$default_time) ^2)






# ============================== Random Forest  ==============================

# Fit Random Forest, we run the algorithm in classification mode
random_forest <- randomForest(as.factor(default_time) ~ ., data = train_data)

# Get predictions for the testing set
predicted_rf <- predict(random_forest, data = test_data)


random_forest$importance










# ============================== LASSO Regression  ==============================


# Run cross validation to find best lambda
best_lambda <- cv.glmnet(x = train_data[ ,-17] %>% as.matrix, y = train_data[ ,17] , alpha = 1, family = "binomial")$lambda.min

# Fit LASSO using lambda from cross validation. We select family = binomial for logistic regression
log_lasso <- glmnet(x = train_data[ ,-17] %>% as.matrix, y = train_data[ ,17] , alpha = 1, family = "binomial", lambda = best_lambda)










