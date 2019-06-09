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

# Add age of position
dat$age <- dat$time- dat$orig_time








# ============================== Initialize Functions  ==============================

# Function to conveniently fit neural network. We will use this later on to evaluate a large number of layer / neuron combinations
fit_nn <- function(layers = c(1,1), train_data1 = train_data, test_data1 = test_data, tr = 0.02, create_network = F){
  
  # Fit neural network
    #fit network
  nn <- try(neuralnet(default_time ~ ., data = train_data1, hidden = layers, act.fct = "logistic", linear.output = F,stepmax = 1e+07,
                      err.fct = "sse", lifesign = "full", threshold = tr, algorithm = "sag", learningrate.factor = list( minus = 0.5, plus = 1.2)))

  if(create_network == F){
    # Compute MSE from probabilities
    return(((predict(nn, test_data1) - test_data$default_time) ^2) %>% mean)
  }else{
    return(nn)
  }
  
}

# Shuffle data and create training and testing set on the go
shuffle <- function(n = 5000, data = scaled_data, ratio = 2/3, check = T){
  

  # Check function call
  if(check != "none") check <- readline(prompt="Are you sure you want to redraw your testing and training data? [y/n] ")
  
  if(check %in% c("no","n")) return(NULL)
  
  if(check %in% c("yes","y","none")){
    
    # Sample n data at random
    temp_data <- data[sample(1:nrow(data),n),]
    
    # Select ratio of the data as trainign data
    index <- sample(1:nrow(temp_data), (nrow(temp_data)*ratio) %>% ceiling)
    
    # Assign training data in global environment
    train_data <<- temp_data[index,][,-c(1:4,22,23)]
    
    # Assign testing data in global environment
    test_data <<- temp_data[-index, ][,-c(1:4,22,23)]
  }
  
}

# Function to create prediction evaluation matrix
prediction_matrix <- function(predictions, observations = test_data$default_time, tr = 0.5){
  
  #transform to binary response
  predictions <- ifelse(predictions < tr, 0,1)
  observations <- ifelse(observations < tr, 0,1)
  
  #get accordances
  nn <- which(predictions == 0) %in% which(observations == 0) %>% sum
  oo <- which(predictions == 1) %in% which(observations == 1) %>% sum
  no <- which(predictions == 0) %in% which(observations == 1) %>% sum
  on <- which(predictions == 1) %in% which(observations == 0) %>% sum
  
  matrix(c(oo,on,no,nn), nrow = 2, byrow = T) %>% set_rownames(1:2) %>% set_colnames(1:2)
  
}

# Function to evaluate performance of models
evaluate_model <- function(model, observations = test_data$default_time, best_lamb = best_lambda, tr = 0.5){
 
  #get predictions from the model
  if(grepl("lasso", deparse(substitute(model))) ){
    predictions <- predict(model, newx = test_data[ ,-17] %>% as.matrix, s = best_lamb, type = "response")
    
  }else{  
    predictions <- predict(model, test_data, type = "response")
  }
  
  #transform to binary response
  predictions_bin <- ifelse(predictions < tr, 0,1)
  
  mat <- prediction_matrix(predictions)
  detected_defaults <- mat[1,1]/colSums(mat)[1] %>% set_names("Ratio")
  mse_cont <- (predictions-observations) ^2 %>% mean
  mse_bin <- (predictions-observations) ^2 %>% mean
  
  return(list(matrix = mat, detected_defaults = detected_defaults, mse_cont = mse_cont, mse_bin = mse_bin))
}







# ============================== Prepare Data ==============================

# Scale data for neural network, we do not scale id and time
max <-  apply(dat[,-c(1,2)], 2 , max)
min <-  apply(dat[,-c(1,2)], 2 , min)
scaled_data <-  cbind(dat[,c(1,2)], as.data.frame(scale(dat[,-c(1,2)], center = min, scale = max - min)))

# Set seed ro reproducibility
set.seed(99)

# We select a subset of the data and split it into training and testing
shuffle(n = 5000, check = "none")










# ============================== Neural Network  ==============================

# Evaluate a number of possible layer / neuron combinations
eval <- apply(expand.grid(4:10,3:6),1, fit_nn, tr = 0.05)

# Fit best network and save under name "nn". For the fitting, we draw a larger dataset
set.seed(99)
shuffle(n= 50000, check = "none")

nn <- fit_nn(expand.grid(4:10,3:6)[which.min(eval),] %>% unlist, create_network = T, tr = 0.05)



# Get predictions for the testing set
predicted_nn <- predict(nn, test_data, type = "response") 









# ============================== Logistic Regression  ==============================

#We perform logistic regression as benchmark model
log_reg <- glm(default_time ~ ., data = train_data, family = binomial())




# Get predictions for the testing set
predicted_reg <- predict(log_reg, newdata = test_data, type = "response")


# Compare logistic regression and neural network
plot(predicted_nn, predicted_reg)








# ============================== Random Forest  ==============================

# Fit Random Forest, we run the algorithm in classification mode
random_forest <- randomForest(as.factor(default_time) ~ ., data = train_data)




# Get predictions for the testing set
predicted_rf <- predict(random_forest, data = test_data) %>% as.character %>% as.numeric

# How many did we get wrong / right
prediction_matrix(predicted_rf)

# Compare with actual data, did we get something right?
defaults_captured(predicted_rf)

# Importance of variables
random_forest$importance










# ============================== LASSO Regression  ==============================



# Run cross validation to find best lambda
best_lambda <- cv.glmnet(x = train_data[ ,-17] %>% as.matrix, y = train_data[ ,17] , alpha = 1, family = "binomial")$lambda.min


# Fit LASSO using lambda from cross validation. We select family = binomial for logistic regression
log_lasso <- glmnet(x = train_data[ ,-17] %>% as.matrix, y = train_data[ ,17] , alpha = 1, family = "binomial", lambda = best_lambda)





# Get predictions for the testing set
predicted_lasso <- predict(log_lasso, newx = test_data[ ,-17] %>% as.matrix, s = best_lambda, type = "response")


# Compare logistic lasso regression and neural network
plot(predicted_nn, predicted_lasso)

# Compare logistic lasso regression and logistic regression

plot(predicted_reg, predicted_lasso)









# ============================== Comparison  ==============================

# Compare the performance of the differnet methods. We evaluate at increasingly large datasets

evaluate_model(nn)
evaluate_model(log_reg)
evaluate_model(log_lasso)


# Reshuffle and draw larger testing data
shuffle(n = 300000, ratio = 1/100)

# In large samples, the NN performs much better
evaluate_model(nn)
evaluate_model(log_reg)
evaluate_model(log_lasso)


# Reshuffle and draw larger testing data
shuffle(n = 500000, ratio = 1/100)

# The fir seems to be stable, with a stable correct prediction rate
evaluate_model(nn)
evaluate_model(log_reg)
evaluate_model(log_lasso)


