# *************************************************************** #
# Machine Learning              : Group Project                   #
# Code File                     : Main                            #
# Student IDs                   : 15605017  - 13614011            #
# *************************************************************** #


# SUMMARY #

# This codes uses a dataset holding information about a mortgage portfolio comprised of 600 000 observations from US banks. The goal is to
# to predict the probability of default using a neural network. As a benchmark model we use simple logistic regression, a widely used 
# technique to estimate PDs. Furthermore we compre our results with those from a LASSO regression and a random forest.




# ============================== Library Calls  ==============================

toload <- c("magrittr","plyr","reshape2","neuralnet","randomForest","glmnet", "caret", "rlist", "tidyr", "mboost")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
sapply(toinstall, install.packages, character.only = TRUE)
sapply(toload, require, character.only = TRUE)






# ============================== Read Data  ==============================

## Working Directory Setting
wd <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(wd)
set.seed(100)
## Read Data from CSV
dat <- read.csv(paste0(wd, "/Data/mortgage.csv")) %T>% attach
## Remove NA observations
dat <- dat[complete.cases(dat),]
## Add age of position
dat$age <- dat$time- dat$orig_time

# How many observations do we want to use for evaluation of the different techniques? Samll number for higher speed of convergence
n_run <- 5000





# ============================== Initialize Functions  ==============================

# Function to conveniently fit neural network. We will use this later on to evaluate a large number of layer / neuron combinations
fit_nn <- function(layers = c(1,1), train_data1 = train_data, test_data1 = test_data, tr = 0.02, create_network = F){
  
  # Fit neural network
    #fit network
  nn <- try(neuralnet(default_time ~ ., data = train_data1, hidden = layers, act.fct = "logistic", linear.output = F, stepmax = 1e+07,
                      err.fct = "sse", lifesign = "full", threshold = tr, algorithm = "sag", learningrate.factor = list( minus = 0.5, plus = 1.2)))

  if(create_network == F){
    # Compute MSE from probabilities
    return(((predict(nn, test_data1) - test_data$default_time) ^2) %>% mean)
  }else{
    return(nn)
  }
  
}

# Shuffle data and create training and testing set on the go
shuffle <- function(n = nrow(dat), data = scaled_data, ratio = 2/3, check = T){
  

  # Check function call
  if(check != "none") check <- askYesNo("Are you sure you want to redraw your testing and training data? [y/n] ")
  
  if(check == F) return(NULL)
  
  if(check == T){
    
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

# Function to evaluate performance of models. Always enter models in form of lists
evaluate_model <- function(model=list(...), observations = test_data$default_time, best_lamb = best_lambda, tr = 0.5){ #modelname added
 

  
  n = length(model)
  tables    = list()
  metrics   = list()
  
  #transform to binary response
  for (i in 1:n){
    
    #get predictions from the model
    if(grepl("lasso", deparse(substitute(model))) ){
      predictions <- predict(model, newx = test_data[ ,-17] %>% as.matrix, s = best_lamb, type = "response")
      
    }else{  
      predictions <- as.integer(predict(model[[i]], newdata=test_data))-1#, type = "response")
    }
    
    predictions_bin <- ifelse(predictions < tr, 0,1)
    
    table  <- prediction_matrix(predictions)
    # table  <- table(observations, predictions_bin, dnn = c("Actual defaults", "Predicted defaults"))[c(2,1),c(2,1)] # Create matrix and Change order columns of matrix
    
    TRP    <- table[1,1]/(table[1,1]+table[1,2])   # True Positive Rate (TPR) or sensitivity or recall or hit rate is a measure of how
    # many true positives were identified out of all the positives identified: Ideally, the model is better if we have this closer to one
    TNR    <- table[2,2]/(table[2,2]+table[2,1])   # True Negative Rate (TNR) or specificity is the ratio of true negatives and total
    # number of negatives we have predicted:
    # If this ratio is closer to zero, the model is more accurate.
    ACC    <- (table[1,1]+table[2,2])/(sum(table)) # Accuracy is the measure of how good our model is. It is expected to be closer to 1, if
    # our model is performing well.
    Prec   <- table[1,1]/(table[1,1]+table[2,1])   # Precision is defined as how many selected items are relevant. That is, how many of
    # the predicted ones are actually correctly predicted. If precision is closer to one, we are more accurate in our predictions
    Recall <- table[1,1]/(table[1,1]+table[1,2])   # Recall, on the other hand, tells how many relevant items we selected.
    
    mse_cont    <- (predictions-observations) ^2 %>% mean
    mse_bin     <- (predictions-observations) ^2 %>% mean
    metrics[[i]]<- data.frame(Model = modelname[i], TRP = TRP, TNR=TNR, ACC=ACC, Precision=Prec, Recall=Recall, MSE_cont = mse_cont, MSE_bin=mse_bin)
    tables[[i]] <- table
  }
  
  names(tables) <- sapply(model, function(x) deparse(substitute(x)))
  
  metrics_all   <- do.call(rbind, metrics)
  
  list(tables, metrics_all)
  #return(list(matrix = mat, detected_defaults = detected_defaults, mse_cont = mse_cont, mse_bin = mse_bin))
}

# Feed with input from evaluate_model
plot_evaluation <- function(evaluate_model_object){ # Insert valuation metrics
  
  metrics_all <- tail(evaluate_model_object,1)
  df <- gather(metrics_all[[1]][, 1:5], key="metric", value="value", -Model) # Bringing in correct shape to feed into ggplot
  ggplot(data=df, aes(x=Model, y=value, color=Model, fill=Model))+ #https://www.r-graph-gallery.com/48-grouped-barplot-with-ggplot2/
    geom_bar(stat="identity")+
    facet_wrap(~metric)+
    theme_light()
}







# ============================== Prepare Data ==============================

# Scale data for neural network, we do not scale id and time
max <-  apply(dat[,-c(1,2)], 2 , max)
min <-  apply(dat[,-c(1,2)], 2 , min)
scaled_data <-  cbind(dat[,c(1,2)], as.data.frame(scale(dat[,-c(1,2)], center = min, scale = max - min)))

# Set seed ro reproducibility

# We select a subset of the data and split it into training and testing
shuffle(n = n_run, check = "none")









# ============================== Neural Network  ==============================

# Evaluate a number of possible layer / neuron combinations
eval <- apply(expand.grid(4:10,3:6),1, fit_nn, tr = 0.05)

# Fit best network and save under name "nn". For the fitting, we draw a larger dataset
set.seed(99)
shuffle(n= n_run, check = "none")

nn <- fit_nn(expand.grid(4:10,3:6)[which.min(eval),] %>% unlist, create_network = T, tr = 0.05)



# Get predictions for the testing set
evaluate_model(list(nn), modelname = c("Neural Network")) 









# ============================== Logistic Regression  ==============================

#We perform logistic regression as benchmark model
log_reg <- glm(default_time ~ ., data = train_data, family = binomial())




# Get predictions for the testing set
predicted_reg <- predict(log_reg, newdata = test_data, type = "response")


# Compare logistic regression and neural network
#plot(predicted_nn, predicted_reg)
evaluate_model(list(log_reg, nn), modelname = c( "Logistic Regression", "Neural Network"))






# == Alternative: Boosted Logistic Regression ==

# Here we use a very handy package to perform k-fold cross validation. The package requires its own training and testing sample splitting procedure
intrain <- createDataPartition(y=dat$default_time, p=2/3, list=FALSE) # splits the data according to y argument and ensures that calls distribution of the data remains
str(intrain) # Output is a set of integers for the rows of dat that belong to the training set
train_data <- dat[ intrain,]
test_data  <- dat[-intrain,]

# initialize cross validation Folds. Here we use 5-fold cross validation, repeated 3 times
fitControl <- trainControl(method="repeatedcv", number = 5, repeats = 3)  

# Fit boosted logistic/probit regression
logitboost_fit <- caret::train(as.factor(default_time) ~ ., data=train_data, method="LogitBoost", trControl = fitControl)

# Looking at how accuracy increases over the training procedure
trellis.par.set(caretTheme())
plot(logitboost_fit) 

# 
evaluate_model(list(log_reg, nn, logitboost_fit), modelname = c("Logistic Regression", "Neural Network", "LogitBoost"))



# ============================== Random Forest  ==============================

# Fit Random Forest, we run the algorithm in classification mode
random_forest <- randomForest(as.factor(default_time) ~ ., data = train_data)




# Get predictions for the testing set
predicted_rf <- predict(random_forest, data = test_data) %>% as.character %>% as.numeric

# How many did we get wrong / right
prediction_matrix(predicted_rf)

# Compare with actual data, did we get something right?
# defaults_captured(predicted_rf)

# Importance of variables
random_forest$importance
evaluate_model(list(log_reg, nn, logitboost_fit, random_forest), modelname = c("Logistic Regression", "Neural Network", "LogitBoost", "untrained RF"))


#=============================== Alternative Way to get Random Forrest via cross validation =========================

rf_fit     <- caret::train(as.factor(default_time) ~ ., data=train_data, method="ranger", trControl = fitControl) # We use "ranger" method for random forrest. The training algorithm seeks to optimize accuracy.

# Looking at how accuracy increases over the training procedure
trellis.par.set(caretTheme())
plot(rf_fit) 

# Lets look how it performs on the test sample.
evaluate_model(list(log_reg, nn, logitboost_fit, random_forest, rf_fit), modelname = c("Logistic Regression", "Neural Network", "LogitBoost", "untrained RF", "trained RF"))





# ============================== LASSO Regression  ==============================



# Run cross validation to find best lambda. Alpha = 1 gives lasso. ALpha = 0 gives ridge
best_lambda <- cv.glmnet(x = train_data[ ,-17] %>% as.matrix, y = train_data[ ,17] , alpha = 1, family = "binomial")$lambda.min


# Fit LASSO using lambda from cross validation. We select family = binomial for logistic regression
log_lasso <- glmnet(x = train_data[ ,-17] %>% as.matrix, y = train_data[ ,17] , alpha = 1, family = "binomial", lambda = best_lambda)





# Get predictions for the testing set
predicted_lasso <- predict(log_lasso, newx = test_data[ ,-17] %>% as.matrix, s = best_lambda, type = "response")


# Compare logistic lasso regression and neural network
plot(predicted_nn, predicted_lasso)

# Compare logistic lasso regression and logistic regression

plot(predicted_reg, predicted_lasso)




# ============================== glmboost  ==============================

glm_fit     <- caret::train(as.factor(default_time) ~ ., data=train_data, method="glmboost", trControl = fitControl) # We use "ranger" method for random forrest. The training algorithm seeks to optimize accuracy.

# Looking at how accuracy increases over the training procedure
plot(glm_fit) 

# Lets look how it performs on the test sample.
metrics <- evaluate_model(list(log_reg, nn, logitboost_fit, random_forest, rf_fit, glm_fit), modelname = c("Logistic Regression", "Neural Network", "LogitBoost", "untrained RF", "trained RF", "GLM Boost"))
metrics 
plot_evaluation(metrics)


there_are_no_errors <- F
if(there_are_no_errors){
  # ============================== Comparison  ==============================
  
  # Compare the performance of the differnet methods. We evaluate at increasingly large datasets
  
  evaluate_model(nn, modelname = "nn")
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
  
  # The fit seems to be stable, with a stable correct prediction rate
  evaluate_model(nn)
  evaluate_model(log_reg)
  evaluate_model(log_lasso)
}

