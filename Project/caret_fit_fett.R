load("set_up_slave.Rdata")
models_to_run <- list("rf", "xgbDART", "svmRadial", "mxnetAdam")

# Set up cluster for parallel computing during CV
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

# shuffle(6000)

# Carry out model fitting using CV
caret_fit <- lapply(models_to_run, function(x) caret::train(make.names(default_time) ~ ., 
                                                            data=train_data, method= x, trControl = fitControl, metric = "ROC") )

# Stop Cluster

stopCluster(cl)


# Renaming the object to have a better overview
modelnames <- unlist(models_to_run)
names(caret_fit) <-  modelnames

# Save the object
saveRDS(caret_fit, "caret_fit_complex.rds")


# Set up cluster for parallel computing during CV
sub_methods <- c("up", "down", "rose", "smote") # Names of the down/upsampling methods

cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

# Fit model for every type of resampling
subsampled_fits <- lapply(sub_methods, function(x) subsamples(x))

stopCluster(cl)

# Give names for better overview
names(subsampled_fits) <- sub_methods

saveRDS(subsampled_fits, "subsampled_fits_complex.rds")

