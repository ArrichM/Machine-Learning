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

toload <- c("magrittr","plyr","reshape2","neuralnet","randomForest","glmnet", "caret", "rlist", "tidyr", "mboost","dplyr","DMwR","ROSE")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
sapply(toinstall, install.packages, character.only = TRUE)
sapply(toload, require, character.only = TRUE)






# ============================== Read Data  ==============================

## Working Directory Setting
wd <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(wd)
set.seed(100)
## Read Data from CSV

# zipF<- paste0(wd, "/Data/mortgage_csv.rar")
# unzip(zipF,exdir=outDir)
# dat <- read.csv(paste0(wd, "/Data/mortgage.csv"))



# ============================== Initialize Functions  ==============================

# Function to extarct and append lags to the dataframe
get_lags <- function(c_id = 1, data = dat, lag = 2, lookup = c(6,7,8,9,10,11)){
  
  #get observations for obligor c_id
  data <- data[which(data$id == c_id),]
  
  #check wheather enough observations are available
  if(length(data$time) < lag+1) return( matrix(NA, ncol = ncol(data) + length(lookup) * lag) )
  
  #function to extract lags and append to row
  bind_lags <- function(t){
    cbind(data[which(data$time == data$time[t]),],
          apply(data[which(data$time %in%  data$time[(t-1):(t-lag)] ), lookup],2, function(x) as.character(x) %>% as.numeric) %>% matrix(ncol = length(lookup)*lag) )
  }
  
  #extract lags and append to row for each observed time
  out <- sapply(1:length(data$time) %>% tail(n = -3) ,bind_lags) %>% t
  
  print(c(c_id, ncol(out)))
  return(out)
  
}

# Shuffle data and create training and testing set on the go. Also we do a number of manipulations to improve performance
shuffle <- function(n = nrow(dat), data = dat, ratio = 2/3, lags = NULL, unwanted = c(1:2,22,23), col_to_lag = c(6,7,8,9,10,11)){
  
  # Sample n data at random
  if(is.null(lags) == T) temp_data <- data[sample(1:nrow(data),n),]
  
  # Add lags if desired
  if(is.null(lags) == F){
    
    #append lags
    temp_data <- lapply(sample(unique(data$id),n), get_lags, lag = lags, lookup = col_to_lag, data = data) %>% do.call(what = rbind) %>% as.matrix
    
    #transform to numeric
    temp_data <- apply(temp_data, 2 , function(x) as.character(x) %>% as.numeric)
    
    #remove NA
    temp_data <- temp_data[complete.cases(temp_data),] %>% set_colnames( c(colnames(dat),
                                                                           paste0(rep(colnames(dat)[col_to_lag], each = lags),"-L",rep(1:lags,length(col_to_lag))) ) )
    
    
  }
  
  # Remove unwanted columns
  temp_data <- temp_data[,-unwanted]
  
  # Scale data for neural network, we do not scale id and time
  min <- apply(temp_data, 2 , min)
  max <- apply(temp_data, 2 , max)
  
  temp_data <-  as.data.frame(scale(temp_data, center = min, scale = max - min))
  
  # Select ratio of the data as trainign data
  index <- sample(1:nrow(temp_data), (nrow(temp_data)*ratio) %>% ceiling)
  
  # Assign training data in global environment
  train_data <<- temp_data[index,]
  
  # Assign testing data in global environment
  test_data <<- temp_data[-index, ]
  
  
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
evaluate_model <- function(model=list(...), modelname, observations = test_data$default_time, best_lamb = best_lambda, tr = 0.5){ #modelname added
  
  
  
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
  
  names(tables) <- modelname
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

## Remove NA observations
dat <- dat[complete.cases(dat),]

# Add age of position
dat$age <- dat$time- dat$orig_time

# We select n obligors at random and append t lags to the dataset in order to account for time effects
shuffle(n = 3000)


# Very dirty workaround for the column selectino problem
# write.table(test_data, file = "temp.csv")
# test_data <- read.table("temp.csv")
#
# write.table(train_data, file = "temp.csv")
# train_data <- read.table("temp.csv")








# ============================== Neural Network  ==============================



# Fit neural network with 7 neurons in one layer
# nn <- neuralnet(default_time ~ ., data = train_data, hidden = 11, act.fct = "logistic", linear.output = F, stepmax = 1e+07,
#                 err.fct = "sse", lifesign = "full", threshold = 0.05, algorithm = "sag", learningrate.factor = list( minus = 0.5, plus = 1.2))

# Get predictions for the testing set
# evaluate_model(list(nn), modelname = c("Neural Network"))









# ============================== Logistic Regression  ==============================


#We perform logistic regression as benchmark model
log_reg <- glm(default_time ~ ., data = train_data, family = binomial())

# Get predictions for the testing set
predicted_reg <- predict(log_reg, newdata = test_data, type = "response")

# Compare logistic regression and neural network
evaluate_model(list(log_reg), modelname = c( "Logistic Regression"))





# == Alternative: Boosted Logistic Regression ==


# initialize cross validation Folds. Here we use 5-fold cross validation, repeated 3 times
#fitControl <- trainControl(method="repeatedcv", number = 5, repeats = 3,  sampling = "smote", classProbs = TRUE)
fitControl <- trainControl(method="repeatedcv", number = 5, repeats = 5, sampling = "smote", classProbs = TRUE,
                           summaryFunction=twoClassSummary, 
                           savePredictions = T)

# Fit boosted logistic/probit regression
logitboost_fit <- caret::train(make.names(default_time) ~ ., data=train_data, method="LogitBoost", trControl = fitControl, metric="ROC")

# Check probabilities
predict(logitboost_fit, newdata=test_data, type="prob")

# Looking at how accuracy increases over the training procedure
trellis.par.set(caretTheme())
plot(logitboost_fit)

#
evaluate_model(list(logitboost_fit), modelname = c("LogitBoost"))
confusionMatrix(logitboost_fit)












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





# == Alternative Way to get Random Forrest via cross validation ==

rf_fit <- caret::train(make.names(default_time) ~ ., data=train_data, method="ranger", trControl = fitControl, metric ="ROC") # We use "ranger" method for random forrest. The training algorithm seeks to optimize accuracy.

# Looking at how accuracy increases over the training procedure
trellis.par.set(caretTheme())
plot(rf_fit)

# Lets look how it performs on the test sample.
evaluate_model(list(rf_fit), modelname = c("trained RF"))
confusionMatrix(rf_fit)













# ============================== LASSO Regression  ==============================


shuffle(n = 300, lags = 3)

# Run cross validation to find best lambda. Alpha = 1 gives lasso. ALpha = 0 gives ridge
best_lambda <- cv.glmnet(x = train_data[ ,-19] %>% as.matrix, y = train_data[ ,19] , alpha = 1, family = "binomial")$lambda.min


# Fit LASSO using lambda from cross validation. We select family = binomial for logistic regression
log_lasso <- glmnet(x = train_data[ ,-19] %>% as.matrix, y = train_data[ ,19] , alpha = 1, family = "binomial", lambda = best_lambda)

log_lasso$beta


# Get predictions for the testing set
predicted_lasso <- predict(log_lasso, newx = test_data[ ,-19] %>% as.matrix, s = best_lambda, type = "response")










# ============================== glmboost  ==============================


glm_fit     <- caret::train(make.names(default_time) ~ ., data=train_data, method="glmboost", trControl = fitControl)

# Looking at how accuracy increases over the training procedure
plot(glm_fit)

# Lets look how it performs on the test sample.
evaluate_model(list(glm_fit), modelname = c("GLM Boost"))
confusionMatrix(glm_fit)

metrics <- evaluate_model(list(log_reg, logitboost_fit, random_forest, rf_fit, glm_fit), modelname = c("Logistic Regression", "LogitBoost", "untrained RF", "trained RF", "GLM Boost"))
metrics
plot_evaluation(metrics)










# ============================== Multinomial  ==============================

multinomial     <- caret::train(make.names(default_time) ~ ., data=train_data, method="multinom", trControl = fitControl) # We use "ranger" method for random forrest. The training algorithm seeks to optimize accuracy.
predict(mxnet, test_data, type="prob")
evaluate_model(list(multinomial), modelname = c("multinomial"))











# ============================== avNNet  ==============================

avNNet     <- caret::train(make.names(default_time) ~ ., data=train_data, method="avNNet", trControl = fitControl) 
evaluate_model(list(avNNet), modelname = c("ageraged NN"))


# plot
metrics <- evaluate_model(list(logitboost_fit, rf_fit, glm_fit, multinomial, avNNet), modelname = c("LogitBoost",  "trained RF", "GLM Boost", "multinomial", "averaged NN"))
metrics
plot_evaluation(metrics)











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