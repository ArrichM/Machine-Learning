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

toload <- c("magrittr","plyr","reshape2","neuralnet","randomForest","glmnet", "caret", "rlist", "tidyr", "mboost","dplyr","DMwR","ROSE","doParallel", "corrplot", "pROC")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
sapply(toinstall, install.packages, character.only = TRUE)
sapply(toload, require, character.only = TRUE)






# ============================== Read Data  ==============================

rm(list=ls())

## Working Directory Setting
dirname(rstudioapi::getSourceEditorContext()$path) %>% setwd

set.seed(100)
## Read Data from CSV
dat <- read.csv("Data/mortgage.csv")




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


  # Change levels of default data from 0 and 1 to non-default and default. Needed for input in care functions
  temp_data$default_time <-  as.factor(ifelse(temp_data$default_time==0,
                                              "non-default",
                                              "default" ))#, unique = T)

  levels(temp_data$default_time) <- make.names(levels(factor(temp_data$default_time)))

  # Select ratio of the data as trainign data
  index <- sample(1:nrow(temp_data), (nrow(temp_data)*ratio) %>% ceiling)


  # Assign training data in global environment
  train_data <<- temp_data[index,]

  # Assign testing data in global environment
  test_data <<- temp_data[-index, ]


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
      predictions <- predict(model[[i]], newdata=test_data)#, type = "response")
    }
    #predictions_bin <- ifelse(predictions < tr, 0,1)

    table  <- confusionMatrix(predictions, test_data$default_time)$table
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

    #mse_cont    <- (predictions-observations) ^2 %>% mean
    #mse_bin     <- (predictions-observations) ^2 %>% mean
    metrics[[i]]<- data.frame(Model = modelname[i], TRP = TRP, TNR=TNR, ACC=ACC, Precision=Prec, Recall=Recall) #MSE_cont = mse_cont, MSE_bin=mse_bin)
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

# Evaluating ROC
test_roc <- function(model) {
  preds <- predict(model, test_data, type="prob")
  roc_obj <- roc(preds[["default"]],
                 response= test_data$default_time,
                 levels=rev(levels(test_data$default_time)))

  ci(roc_obj)
}

# Function plotting of ROCs

ROC_plot <- function(caret_fit, modelname=models_to_run){
  models_to_run = unlist(modelname)
  preds <- lapply(caret_fit, function(x) predict(x, test_data, type="prob"))
  roc   <- lapply(1:length(models_to_run), function(x) roc(preds[[x]][["default"]],
                                                           response= test_data$default_time,
                                                          levels=rev(levels(test_data$default_time))))

  plot.roc(roc[[1]], auc.polygon=TRUE,
           grid=c(0.1, 0.2), grid.col=c("grey", "red"),
           print.thres=TRUE,
           reuse.auc=FALSE, main="ROC curves")#, #add = TRUE)
  legend("bottomright", legend=modelnames,
         col=c(par("fg"), "blue"), lwd=2)
  for (i in 2:length(modelnames)){
    plot.roc(roc[[i]], add=T)

  }


}

# Plotting function for histograms
hist_plot <- function(caret_fit.=caret_fit, modelname=models_to_run){
  models_to_run = unlist(modelname)
  preds <- lapply(caret_fit, function(x) predict(x, test_data, type="prob"))
  list <- list()
  for (i in 1:length(models_to_run)){
    #par(mfrow=c(1,1))
    list[[i]] <- histogram(~preds[[i]][["default"]]|test_data$default_time,xlab="Probability of Poor Segmentation", main=modelnames[i])
  }
  return(list)
}

# Densityplots
densityplots_all <- function(caret_fit.=caret_fit){
  list=list()
  for (i in 1:length(caret_fit)){
    list[[i]] <- densityplot(caret_fit[[i]], pch = "|", main=modelnames[i])
  }
  trellis.par.set(caretTheme())
  return(list)
}


# ============================== Prepare Data ==============================

## Remove NA observations
dat <- dat[complete.cases(dat),]

# Add age of position
dat$age <- dat$time- dat$orig_time

# Undersample non-defaults from the whole dataset. Holdout testing data right away
shuffle(n=5000, ratio = 3/4)

# Do undersampling of nondefaults
# def_data <- train_data[which(train_data$default_time == "default"),]
# liv_data <- train_data[which(train_data$default_time == "non.default"),]
# train_data <- rbind(def_data,liv_data[sample(1:nrow(liv_data),nrow(def_data)),])





# ============================== Run different caret models ==============================


# Create fit constrol object which will control all models. We balance our dataset using the smote algortihm
fitControl <- trainControl(method="repeatedcv", number = 5, repeats = 3, classProbs = TRUE,
                           summaryFunction=twoClassSummary, sampling = "down",
                           savePredictions = T)

# We specify the desired models
models_to_run <- list("LogitBoost","glmboost","multinom","avNNet")


# Set up cluster for parallel computing during CV
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

# shuffle(6000)

# Carry out model fitting using CV
caret_fit <- lapply(models_to_run, function(x) caret::train(make.names(default_time) ~ .,
                                          data=train_data, method= x, trControl = fitControl, metric = "ROC") )

# Stop Cluster

stopCluster(cl)

# Rename
modelnames <- unlist(models_to_run)
names(caret_fit) <-  modelnames

# Check and compare metrics
metrics <- evaluate_model(caret_fit, modelname = modelnames) %T>% print


# ====================================== Plots ========================================
trellis.par.set(caretTheme())
# Plot rocs
ROC_plot(caret_fit)

# Plot densities
densityplots_all()

#Plot historgrams
hist_plot()



# Comparing Multiple Models
# Having set the same seed before running gbm.tune and xgb.tune
# we have generated paired samples and are in a position to compare models
# using a resampling technique.
# (See Hothorn at al, "The design and analysis of benchmark experiments
# -Journal of Computational and Graphical Statistics (2005) vol 14 (3)
# pp 675-699)


# ================================== Performances ==============================


rValues <- resamples(caret_fit)
rValues$values
summary(rValues)


# Load theme
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(rValues, layout = c(3, 1))

#Plot
bwplot(rValues,metric="ROC",main="ROC")	# boxplot
dotplot(rValues,layout = c(3, 1))	# dotplot
splom(rValues,metric="ROC")
xyplot(rValues, what = "BlandAltman")

# Quick check if there is difference between models
difValues <- diff(rValues)
summary(difValues)

trellis.par.set(theme1)
bwplot(difValues, layout = c(3, 1))

# Logit boosts seems to perform significantly worse
trellis.par.set(caretTheme())
dotplot(difValues)


# ================================= Performances with Subsampling =========================

#Function to Fit caret_fit for every type of resampling (upsampling, downsampling, SMOTE, ROSE)

subsamples <- function(char){
  fitControl$sampling <- char
  fit <- lapply(models_to_run, function(x) caret::train(make.names(default_time) ~ .,
                                                              data=train_data, method= x, trControl = fitControl, metric = "ROC") )
  names(fit) <- modelnames
  return(fit)
}

# Set up cluster for parallel computing during CV
sub_methods <- c("up", "down", "rose", "smote") # Names of the down/upsampling methods

cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

# Fit model for every type of resampling
subsampled_fits <- lapply(sub_methods, function(x) subsamples(x))

stopCluster(cl)


names(subsampled_fits) <- sub_methods



# Extract for each subsampling method the corresponding method
all_resamples <- list()
all_tests <- list()
for(j in 1:length(modelnames)){
  models <- list()
  for(i in 1:length(sub_methods)){
    mod <- subsampled_fits[[sub_methods[i]]][[modelnames[j]]]
    models[[i]] <- mod
  }
  names(models)   <- sub_methods
  all_resamples[[j]] <- resamples(models)
  all_tests[[j]]      <- lapply(models, function(x) test_roc(x))
  all_tests[[j]]      <- lapply( all_tests[[j]], as.vector)
  all_tests[[j]]      <- do.call("rbind", all_tests[[j]])
  colnames(all_tests[[j]]) <- c("lower", "ROC", "upper")
  all_tests[[j]] <- as.data.frame(all_tests[[j]])
}

# Reassign themodelnames
names(all_resamples) <- modelnames
names(all_tests) <- modelnames

# Get summary for all resamples
summary <- lapply(all_resamples, function(x) summary(x, metric = "ROC"))

# Plot them
bwplot_subsample_list <- list()
for (i in 1:length(modelnames)){
  trellis.par.set(theme1)
  bwplot_subsample_list[[i]] <-bwplot(all_resamples[[modelnames[i]]], main=modelnames[i], layout=c(3,1))

}

bwplot_subsample_list

# Now we are testing wether there is a significant difference in prediction power between the models
dif_subsample_plot <- list()
for (i in 1:length(modelnames)){
  difsubsampling_method <- diff(all_resamples[[i]])
  trellis.par.set(theme1)
  dif_subsample_plot[[i]] <-dotplot(difsubsampling_method, main=modelnames[i])

}
dif_subsample_plot

# No real difference from zero

all_tests
