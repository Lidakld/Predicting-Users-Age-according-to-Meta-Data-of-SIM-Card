setwd("~/Predicting-User-Age-by-Mobile-Phone-Meta-Data")
data <- read.csv("cs597_project/data/feature_reduced_data.csv", header=TRUE, sep=",")
summary(data)
original <- data

#call the mice package
library(mice)
init = mice(data, maxit=0) 
<<<<<<< HEAD
=======
imp <- mice(data, print=F)
plot(imp)

>>>>>>> b86754a17a3473b7fef5c43a7361c6788b1ece65
meth = init$method
predM = init$predictorMatrix

#specify the methods for imputing the missing values. As all the values are continuous, we shall select norm.
meth[c("RECHARGE_VALUE","RECHARGE_AVG","MIN_RECHARGE_AVG",
       "INSTAGRAM_MB","P2_MADE_CALLS","P2_PERC_DAYS_SOCIAL_NETWORK_OTT","P4_PERC_WHATSAPP_MB",
       "PERC_INTERNET_SESSIONS_IN_P1","PERC_RECEIVED_CALLS_DURATION_IN_P2","WEEKDAY_PERC_INTERNET_MB",
       "WEEKDAY_P23_INSTAGRAM_MB","WEEKDAY_P23_PERC_WHATSAPP_SESSIONS","WEEKEND_HOLYDAY_P14_PERC_WHATSAPP_MB",
       "WEEKEND_HOLYDAY_P14_PERC_BANK_SESSIONS","WEEKEND_HOLYDAY_P14_WHATSAPP_MB_BY_SESSION")]="norm" 

#run the multiple (m=5) imputation.
set.seed(103)
<<<<<<< HEAD
imputed = mice(data, method=meth, predictorMatrix=predM, m=5)
=======
imputed = mice(data, method=meth, predictorMatrix=predM, m=3.5)
>>>>>>> b86754a17a3473b7fef5c43a7361c6788b1ece65

#Create a dataset after imputation.
imputed <- complete(imputed)

#Check for missings in the imputed dataset.
sapply(imputed, function(x) sum(is.na(x)))


# Write CSV in R
write.csv(imputed, file = "cs597_project/data/preprocessed_data.csv")
