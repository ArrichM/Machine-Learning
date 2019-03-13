###################################
# Machine Learning - Homework 3   #
# Maximilian Arrich #
# Florian Benkhalifa #
###################################

#v2

#prepare workspace
rm(list=ls())
library(MASS)
library(magrittr)

attach(Boston)

######Exercise 1.1
#We define a function to perform stochastic gradient descent using variable numbers of epochs and batch size. 
#Batch size = 1 is equivalent to simple stochastic gradient descent, batch size = length(y) is full gradient descent

stochastic_descent <- function(y = medv, x = lstat, beta0 = 20, beta1 = 0, eta = 0.000009, epochs = 20, batch_size = 1, seed = 100, trace = F){
  
  #set seed to ensure reproducibility
  set.seed(seed)
  
  #how many obervations do we have available for fitting?
  available <- min(length(x),length(y))
  
  #create containers
  R2 <- rep(NA, epochs)
  RSE <-  rep(NA, epochs)
    
  #create first loop representing one full epoch each
  for(j in 1:epochs){
    
    #shuffle data to randomize learning path
    shuffle  <- sample(1:available)
    x <- x[shuffle]
    y <- y[shuffle]
    
    #how many full batches can we get?
    number_of_batches <- floor(available/batch_size)
    #create spliting factors accordindg to number of batches and batch size
    splitting_factors  <- rep(1:number_of_batches, batch_size)
    #create list of indices used for selecting the batches below
    batch_selector <- split(1:(batch_size*number_of_batches), splitting_factors)
  
    #start iteration according to the lecture slides
    for(i in batch_selector){
      
      #obtain predictions for y
      yhat <- beta0 + beta1*x[i]
      #use prediction to obtain error terms
      error1 <-sum(y[i] - yhat) #simple error
      error2 <- sum((y[i] - yhat)*x[i]) #weihghed error
    
      #use formuzlas to recalibrate betas and start next iteration
      beta0 <- beta0 + 2*eta*error1
      beta1 <- beta1 + 2*eta*error2
    }
    
    #calculate error measures according to the lecture slides
    RSS <- sum((y - (beta0 + beta1*x))^2)
    RSE[j] <- sqrt(RSS/(available-2))
    R2[j] <- (sum((y - mean(y))^2) - sum((y-(beta0 + beta1*x))^2)) / sum((y - mean(y))^2)
    
  }
  if(trace == F) RSE <- tail(RSE,1); R2 <- tail(R2,1)

  return(list(cbind(beta0,beta1)%>%set_colnames(c("beta0","beta1")),
              cbind(RSE,R2)%>%set_colnames(c("RSE","R2"))) %>% set_names(c("betas","errors")))
}

#run function using batch size = 1 which is simple stochastic descent
stochastic_descent(trace = T)

stochastic_descent(eta = 0.0009, epochs = 50,trace = F)

lm(medv~lstat)


######Exercise 1.2
#run function using batch size = 32 wich is mini batch stochastic descent
stochastic_descent(batch_size = 32,epochs = 20)




######Exercise 1.3
#we plot the RSEs for each epoch sewtting batch size equal to 1:
plot(stochastic_descent(trace = T)$errors[,1])




######Exercise 1.3
#we plot the RSEs for each epoch sewtting batch size equal to 32:
plot(stochastic_descent(batch_size = 32, trace = T)$errors[,1])




######Exercise 1.4
#create function to measure elapsed time of function call
measure_time <- function(FUN, ...){
  start <- Sys.time()
  FUN(...)
  end <- Sys.time()
  
  return(as.numeric(end-start))
}

#plot the elapsed time for the stochastic gradient descent running with 1 to 20 epochs
plot(lapply(1:20, function(x)  measure_time(stochastic_descent, epochs = x)) %>% unlist)




######Exercise 1.5
#plot the elapsed time for the stochastic gradient descent  with batch size 32  running with 1 to 20 epochs
plot(lapply(1:20, function(x)  measure_time(stochastic_descent,batch_size = 32, epochs = x)) %>% unlist)

