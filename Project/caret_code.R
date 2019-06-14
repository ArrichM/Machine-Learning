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

toload <- c("magrittr","plyr","reshape2","neuralnet","randomForest","glmnet", "caret", "rlist", "tidyr", "mboost","dplyr","DMwR","ROSE","doParallel", "corrplot", "pROC", "gridExtra", "lattice", "skimr")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
sapply(toinstall, install.packages, character.only = TRUE)
sapply(toload, require, character.only = TRUE)






# ============================== Prepare Data  ==============================

rm(list=ls())
gc()

## Working Directory Setting
dirname(rstudioapi::getSourceEditorContext()$path) %>% setwd

set.seed(100)
## Read Data from CSV
dat <- read.csv("Data/mortgage.csv")

## Remove NA observations
dat <- dat[complete.cases(dat),]

# Add age of position
dat$age <- dat$time- dat$orig_time






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
shuffle <- function(n = nrow(dat), data = dat, ratio = 2/3, lags = NULL, unwanted = c(1:2,22,23), col_to_lag = c(6,7,8,9,10,11), seed = NULL){
  
  if(is.null(seed) == F) set.seed(seed)
  
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





ROC_plot <- function(caret_fit, modelname=models_to_run){
  
  
  models_to_run = unlist(modelname)
  n <- length(models_to_run)
  preds <- lapply(caret_fit, function(x) predict(x, test_data, type="prob"))
  roc   <- lapply(1:length(models_to_run), function(x) roc(preds[[x]][["default"]],
                                                           response= test_data$default_time,
                                                           levels=rev(levels(test_data$default_time))))
  
  plot.roc(roc[[1]], auc.polygon=TRUE,
           grid=c(0.1, 0.2), grid.col=c("grey", "red"),
           print.thres=TRUE,
           reuse.auc=FALSE, main="ROC curves",
           #col = gray((1:length(models_to_run))/6))#, #add = TRUE)
           col = rainbow(n))
  legend("bottomright", legend=models_to_run,
         col = rainbow(length(models_to_run)), lwd=2)
  for (i in 2:length(models_to_run)){
    plot.roc(roc[[i]], add=T,
             col = rainbow(n)[i])
    
  }
}




# Plotting function for histograms
hist_plot <- function(caret_fit.=caret_fit, modelname=models_to_run){
  models_to_run = unlist(modelname)
  preds <- lapply(caret_fit, function(x) predict(x, test_data, type="prob"))
  list <- list()
  par(mfrow=c(10,2))
  for (i in 1:length(models_to_run)){
    #par(mfrow=c(1,1))
    list[[i]] <- histogram(~preds[[i]][["default"]]|test_data$default_time,xlab="Probability of Poor Segmentation", main=models_to_run[i])
  }
  do.call(grid.arrange, c(list, list(nrow=2)))
}




# Densityplots
densityplots_all <- function(caret_fit.=caret_fit){
  list=list()
  for (i in 1:length(caret_fit)){
    list[[i]] <- densityplot(caret_fit[[i]], pch = "|", main=models_to_run[i])
  }
  trellis.par.set(caretTheme())
  do.call(grid.arrange, c(list, list(nrow=2)))
}




# Box and Dotplots
resample_plot <- function(resamples){
  theme1 <- trellis.par.get()
  theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
  theme1$plot.symbol$pch = 16
  theme1$plot.line$col = rgb(1, 0, 0, .7)
  theme1$plot.line$lwd <- 2
  trellis.par.set(theme1)
  grid.arrange(bwplot(resamples, layout = c(3, 1), main="Boxplots"),
               dotplot(resamples,layout = c(3, 1), main="Dotplots"),
               nrow=2)	# dotplot
}




# Plot function for the subsamples to be compared
bwplot_subsampling <- function(sub_resamples){
  bwplot_subsample_list <- list()
  for (i in 1:length(models_to_run)){
    trellis.par.set(theme1)
    bwplot_subsample_list[[i]] <-bwplot(sub_resamples[[models_to_run[i]]], main=models_to_run[i], layout=c(3,1))
    
  }
  
  do.call(grid.arrange, c(bwplot_subsample_list, list(nrow=length(bwplot_subsample_list),
                                                      top=textGrob(gp=gpar(col='black',
                                                                           fontsize=20),"Subsamples"))))
  #top="top label"))))))
}




# Simple plot to display the differences
dif_subsample_plot <- function(all_resample){
  dif_subsample_list <- list()
  for (i in 1:length(models_to_run)){
    difsubsampling_method <- diff(all_resamples[[i]])
    trellis.par.set(theme1)
    dif_subsample_list[[i]] <-dotplot(difsubsampling_method, main=models_to_run[i])
  }
  do.call(grid.arrange, c(dif_subsample_list, list(nrow=length(dif_subsample_list),
                                                   top=textGrob(gp=gpar(col='black',
                                                                        fontsize=20),"Differences"))))
  #top="top label")))
}




# Functin to cope with the multifits and extract relevant info
extract_submodels <- function(subsampled_fits1 = subsampled_fits, models_to_run1 = models_to_run, sub_methods1 = sub_methods){
  
  all_resamples <- list()  # setting up empty list which gets filled with every resample for each model and each subsampling after every iteration. This will be a nested list in the end
  all_tests <- list()     # Setting up empty list which gets filled with roc test for every model and every subsampling method. this will be an empty list in the end
  for(j in 1:length(models_to_run1)){ # creates a list where for every model there are 4 version, corresponding to the subsampling technique
    models <- list()              # create a list of the same model for different subsampling methods
    for(i in 1:length(sub_methods1)){
      mod <- subsampled_fits1[[sub_methods1[i]]][[models_to_run1[j]]]
      models[[i]] <- mod
    }
    names(models)   <- sub_methods1 # Rename
    all_resamples[[j]] <- resamples(models) # Simply apply resampling
    all_tests[[j]]      <- lapply(models, function(x) test_roc(x))
    all_tests[[j]]      <- lapply( all_tests[[j]], as.vector)
    all_tests[[j]]      <- do.call("rbind", all_tests[[j]])
    colnames(all_tests[[j]]) <- c("lower", "ROC", "upper")
    all_tests[[j]] <- as.data.frame(all_tests[[j]])
  }
  
  return(list(all_resamles = all_resamples %>% set_names(models_to_run), 
              all_tests = all_tests %>% set_names(models_to_run)))
}




train_caret <- function(x, control = fitControl, subsave = T){
  
  # Print fitting info
  paste("Currently fitting:", x, "              ","Sampling method:", control$sampling) %>% print
  
  #Save Sys.time for time evaluation
  t1 <- Sys.time()
  
  if(x == "glm"){
    #For fitting the simple logit
    model <- caret::train(default_time ~ .,
                          data=train_data, method= x, trControl = control, metric = "ROC", family = "binomial")
    save(model, file = paste0(x, ".Rdata"))
  }else{
    #For fitting the other models
    model <- caret::train(default_time ~ .,
                          data=train_data, method= x, trControl = control, metric = "ROC")
  }
  #Save to disk so we can work with subresults right away
  if(subsave == T) save(model, file = paste0(x,"_",control$sampling,".Rdata"))
  
  # Print post fitting info
  paste("Done fitting. Time elapsed:",(Sys.time() - t1) %>% round(digits = 2)) %>% print
  
  #Return resulting object
  return(model)
}




# ============================== Data Exploration Tools ==============================

# Function plotting of ROCs
theme1 <- trellis.par.get()

# Warning: if error occurs, try to enlarge the plot device. It should work then.

# Blue boxes represent regions where most data lie inside.
# At first glance gdp_time seems to be important, FICO_orig_time, LTV time and interest rate time
trellis.par.set(theme = col.whitebg(), warn = FALSE)
featurePlot(x = train_data[,-19],
            y = train_data$default_time,
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")))

# Same here, , especially GDP time. We w ill confirm the importances later. These plots just give an idea
# For more interpretations look at https://www.machinelearningplus.com/machine-learning/caret-package/  Chapter 4
featurePlot(x = train_data[,-19],
            y = train_data$default_time,
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")))


# Some summary statistic. Extend console window to see all of it
skimmed <- skim_to_wide(train_data)
skimmed[, c(1:5, 9:11, 13, 15:16)] %>% kable()


# summary statistics by default
train_data[, c(1:5, 9:11, 13, 15:16, 19)] %>% dplyr::group_by(default_time) %>% skim()








# ============================== Run different caret models ==============================






##### First step - Decide over sampling method. For efficiency, we use only 5k observations #####

# Set up sampling methods to evaluate
sub_methods <- c("up", "down", "rose", "smote")

# We specify the desired models
models_to_run <- c("glm", "LogitBoost", "avNNet", "svmRadial", "xgbDART")

# Draw training and testing sample
shuffle(10000, ratio = 2/3)

#Function to Fit caret_fit for every type of resampling (upsampling, downsampling, SMOTE, ROSE)
subsamples <- function(char, models){
  
  fitControl_sub <- fitControl <- trainControl(method="repeatedcv", number = 5, repeats = 2, classProbs = TRUE,
                                           summaryFunction = twoClassSummary, sampling = char,
                                           savePredictions = T)
  
  fit <- lapply(models, function(x) try(train_caret(x, control = fitControl_sub)) )
  names(fit) <- models
  return(fit)
}


# Set up cluster for faster parallel fitting
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

# Fit model for every type of resampling
subsampled_fits <- lapply(sub_methods, subsamples, models = models_to_run)

# Stop CLuster when computing is done
stopCluster(cl)

# Set correct nsmes for the list objects
names(subsampled_fits) <- sub_methods

# Extract for each subsampling method the corresponding method and test right away
test_res <- extract_submodels(subsampled_fits)

# Extract results
all_tests <- test_res$all_tests
all_resamples <- test_res$all_resamles

















##### .Second step - Fit models with chosen sampling method on larger dataset. #####


shuffle(n = 100000, ratio = 2/3)


# Create fit constrol object which will control all models. We balance our dataset using the smote algortihm
fitControl <- trainControl(method="repeatedcv", number = 5, repeats = 2, classProbs = TRUE,
                           summaryFunction = twoClassSummary, sampling = "up",
                           savePredictions = T)


# Set up cluster for faster parallel fitting
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

# Carry out model fitting using CV
caret_fit <- lapply(models_to_run, function(x)  train_caret(x)) %>% set_names(models_to_run)

# Stop CLuster when computing is done
stopCluster(cl)



save(caret_fit, "largefit.Rdata")

# Check and compare metrics. Especially the confusion matrices are of interest
metrics <- evaluate_model(caret_fit, modelname = models_to_run) %T>% print


# ====================================== Plots ========================================


# Check out importance of variables for each of the models. This simply displays how much weights they were given. It can be seen that
# GDP time seems to be important in each of them.
importance <- lapply(caret_fit, varImp)
lapply(importance, plot)


# Plot rocs. The more the lines are at the left top, the better they perform
ROC_plot(caret_fit)

# Comparing Multiple Models
# Having set the same seed before running gbm.tune and xgb.tune
# we have generated paired samples and are in a position to compare models
# using a resampling technique.
# (See Hothorn at al, "The design and analysis of benchmark experiments
# -Journal of Computational and Graphical Statistics (2005) vol 14 (3)
# pp 675-699)

#Plot historgrams. This is also exploration
hist_plot()


# ================================== Comparison between models ==============================

# With resample function one cam make statistical statements about their performance differences
# By resampling we can check the resample distributions.
rValues <- resamples(caret_fit)
rValues$values
summary(rValues)


# Use our prespecified plot function from above
resample_plot(rValues)

#Plot
splom(rValues,metric="ROC")
xyplot(rValues, what = "BlandAltman")

# Since they are fit it makes sense to interfere
# about the differences between the models. We even perform t-test to evaluate the null that there is no
# difference between the models. We do that by applying diff function
difValues <- diff(rValues)
summary(difValues)

# Plot
resample_plot(difValues)

# ================================= Subsampling for class imbalances =========================

# Since we know that our target has minor class representation, we can encounter that problem by down/up sampling and more methods.
# Since our data has less than 3% of defaults, this needs to be handled with care. Therefore, prior to fitting the final model, we
# want to test which of the subsampling methods is superior. Therefore, we fit each model for each method, and the check how they
# perform under each technique. Finally, we choose the best

# ================

# Reassign themodels_to_run for overview. We have a nested list where we can access every model under each subsampling technique
names(all_resamples) <- models_to_run
names(all_tests) <- models_to_run

# Get summary for all resamples
summary <- lapply(all_resamples, function(x) summary(x, metric = "ROC"))

# Plot bw plots of all models and compare them
bwplot_subsampling(all_resamples)

# Now we are testing wether there is a significant difference in prediction power between the models
dif_subsample_plot(all_resamples)
# No real difference from zero

all_tests

