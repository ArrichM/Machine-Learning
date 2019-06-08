# *************************************************************** #
# Machine Learning              : Group Project                   #
# Code File                     : Main                            #
# Student IDs                   : 15605017  - 13614011            #
# *************************************************************** #





# ============================== Library Calls  ==============================

toload <- c("magrittr","plyr","reshape2","neuralnet")
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
temp_data <- scaled_data[sample(1:nrow(scaled_data),50000),]


# Select 2/3 of the data as trainign data
index <- sample(1:nrow(temp_data), (nrow(temp_data)*0.6) %>% ceiling)

train_data <- temp_data[index,][,-c(1:4,22,23)]

test_data <- temp_data[-index, ][,-c(1:4,22,23)]










# ============================== Generate Neural Network  ==============================


# Fit neural network
nn <- neuralnet(default_time ~ ., data = train_data, hidden = 10, act.fct = "logistic", linear.output = F,
                err.fct = "sse", lifesign = "full", threshold = 0.01, algorithm = "sag", learningrate.factor = list( minus = 0.5, plus = 1.2))


# Get predictions for the testing set
predicted_nn = predict(nn, test_data)
predicted_nn_bin <- ifelse(predicted_nn > 0.5, 1,0)

# How many defualts were predicted?
table(predicted_nn_bin)

# Compare with actual data, did we get something right?
compare <- cbind(predicted_nn_bin, test_data$default_time)[which(test_data$default_time == 1),] %T>% print

# How many did we get wrong / right
table(predicted_nn_bin == test_data$default_time)


# Compute MSE from probabilities
mse_nn <- mean((predicted_nn - test_data$default_time) ^2)






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















