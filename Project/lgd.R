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




# Generate Neural Network ------------------------------------------




# Scale data for neural network, we do not scale id and time
max <-  apply(dat[,-c(1,2)], 2 , max)
min <-  apply(dat[,-c(1,2)], 2 , min)
scaled_data <-  cbind(dat[,c(1,2)], as.data.frame(scale(dat[,-c(1,2)], center = min, scale = max - min)))


# We select a subset of the data and split it into training and testing
temp_data <- scaled_data[sample(1:nrow(scaled_data),3000),]

index <- sample(1:nrow(temp_data), (nrow(temp_data)*0.6) %>% ceiling)

train_data <- temp_data[index,][,-c(1,2,22,23)]

test_data <- temp_data[-index, ][,-c(1,2,22,23)]
  

# Fit neural network
nn <- neuralnet(default_time ~ ., data = train_data, hidden = c(7,7), act.fct = "logistic", 
                err.fct = "sse", lifesign = "full", threshold = 0.01, algorithm = "sag", learningrate.factor = list( minus = 0.5, plus = 1.2))


# Get predictions for the testing set
predicted = ifelse(predict(nn, test_data) > 0.5,1,0)
table(predicted)

# Compare with actual data
table(predicted == test_data$default_time)
table(test_data$default_time)


## Logistic regression as benchmark

log_reg <- glm(default_time ~ ., data = train_data, family = "binomial")


pred <- ifelse(fitted(log_reg) > 0.5,1,0)


table(pred == train_data$default_time)
table(dat$default_time)






length(pred)
length(predicted)
nrow(test_data)

table(pred == predicted)


correct <- sum(ifelse(predicted == test_data$default_time,1,0))

false <- sum(ifelse(predicted != test_data$default_time,1,0))

unique(predicted)

false <- sum(test_data$default_time)


nrow(predicted)
length(train_data$default_time)

mse <- sum((test_data$default_time - predicted)^2) / nrow(test_data)















