# *************************************************************** #
# Machine Learning              : Group Project                   #
# Code File                     : Main                            #
# Student IDs                   : 15605017  - 13614011            #
# *************************************************************** #

## Working Directory Setting
wd <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(wd)





# ============================== Library Calls  ==============================

toload <- c("data.table","lubridate","zoo","xts","readxl","readr","ggplot2","pander","tableHTML","stringr","gdata","MASS","copula",
            "mvnmle","tseries","VineCopula","stargazer","psych","gtable","gridExtra","lmtest","scales","latex2exp","ggfortify","magrittr")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
lapply(toinstall, install.packages, character.only = TRUE)
lapply(toload, require, character.only = TRUE)





# ============================== Read Data  ==============================
















