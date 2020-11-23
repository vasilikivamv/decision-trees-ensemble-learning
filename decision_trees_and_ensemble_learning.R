## Data loading -------------------------------------------------------------------------------------------------------------------

mydata <- read.delim ("D:/train_data.txt", header = FALSE, sep = ",")

colnames(mydata) <- c("id","Jitter(local)","Jitter(local,absolute)","Jitter(rap)",
                      "Jitter(ppq5)","Jitter(ddp)","Shimmer(local)","Shimmer(local,db)","Shimmer(apq3)",
                      "Shimmer(apq5)","Shimmer(apq11)","Shimmer(dda)","AC","NTH",
                      "HTN","Median pitch","Mean pitch", "Standard deviation","Minimum pitch",
                      "Maximum pitch","Number of pulses","Number of periods","Mean period",
                      "Standard dev.period","Frac localUnvoicedFrames","NumberVoiceBreaks",
                      "DegreeVoiceBreaks","UPDRS","parkinson")

mydata <- mydata[,-1]              # remove id
mydata <- mydata[,-27]             # remove UPDRS


## Splitting/ Scaling the dataset---------------------------------------------------------------------------------------------------

set.seed(123)                                   # set seed for reproducibility
N <- nrow(mydata)                               # number of rows
train_id <- sample(1:N, N*0.6, replace = F)     # train id
train.set <- mydata[train_id,]                  # training set
test.set <- mydata[-train_id,]                  # test set

## Scale the data

X_train <- scale(train.set[, c(1:26)])
X_test <- scale(test.set[, c(1:26)])
y_train <- train.set$parkinson
y_test <- test.set$parkinson

trn <- data.frame(X_train, y_train)
tst <- data.frame(X_test, y_test)

## Load the working libraries---------------------------------------------------------------------------------------------------------

library(rattle)
library(rpart)             # for classification trees
library(rpart.plot)        # for plotting decision trees
library(vip)               # for feature importance
library(tidyverse)         # for data manipulation
library(caret)             # for machine learning 
library(ggplot2)           # for plots
library(ranger)            # for random forests
library(plotROC)           # for ROC curves
library(ipred)             
library(gbm)               # for GBM
library(xgboost)           # for XGBoost
library(adabag)            # for AdaBoost


## Fully grown Tree---------------------------------------------------------------------------------------------------------------------

## model1 <- rpart(y_train ~., data = trn, method = "class",control = list(cp = 0, xval = 10),minsplit = 0,minbucket=0)
set.seed(123)
model1 <- rpart(y_train ~., data = trn, method = "class", cp=0)

printcp(model1)                                            # display the results

# 
# Classification tree:
#   rpart(formula = y_train ~ ., data = trn, method = "class", cp = 0)
# 
# Variables actually used in tree construction:
#   [1] Frac.localUnvoicedFrames HTN                      Jitter.local.            Jitter.local.absolute.  
# [5] Jitter.rap.              Maximum.pitch            Median.pitch             Minimum.pitch           
# [9] Number.of.pulses         Shimmer.apq11.           Shimmer.apq5.            Shimmer.local.db.       
# [13] Standard.deviation      
# 
# Root node error: 307/624 = 0.49199
# 
# n= 624 
# 
# CP nsplit rel error  xerror     xstd
# 1  0.2019544      0   1.00000 1.03583 0.040677
# 2  0.0488599      1   0.79805 0.90228 0.040427
# 3  0.0260586      2   0.74919 0.83388 0.040023
# 4  0.0228013      7   0.61238 0.87622 0.040296
# 5  0.0195440      9   0.56678 0.85016 0.040137
# 6  0.0146580     13   0.48534 0.84039 0.040070
# 7  0.0130293     15   0.45603 0.83388 0.040023
# 8  0.0097720     18   0.41694 0.82736 0.039975
# 9  0.0032573     24   0.35179 0.81759 0.039899
# 10 0.0021716     25   0.34853 0.83713 0.040047
# 11 0.0000000     28   0.34202 0.84039 0.040070


# plot (The functions for plotting were retrieved from: https://rstudio-pubs-static.s3.amazonaws.com/27179_e64f0de316fc4f169d6ca300f18ee2aa.html)

only_count <- function(x, labs, digits, varlen)
{
  paste(x$frame$n)
}

boxcols <- c("pink", "palegreen3")[model1$frame$yval]

par(xpd=TRUE)
prp(model1, faclen = 0, cex = 0.8, node.fun=only_count, box.col = boxcols)
legend("bottomleft", legend = c("PD negative","PD possitive"), fill = c("pink", "palegreen3"),
       title = "Group",cex = 0.7)


# Make predictions on the train data
predicted.classes <- model1 %>% 
  predict(trn, type = "class")
confusionMatrix(as.factor(predicted.classes), as.factor(trn$y_train))

#Confusion Matrix and Statistics

# Reference
# Prediction   0   1
# 0 245  43
# 1  62 274
# 
# Accuracy : 0.8317       
# 95% CI : (0.8, 0.8603)
# No Information Rate : 0.508        
# P-Value [Acc > NIR] : < 2e-16      
# 
# Kappa : 0.663        
# 
# Mcnemar's Test P-Value : 0.07898      
#                                        
#             Sensitivity : 0.7980       
#             Specificity : 0.8644       
#          Pos Pred Value : 0.8507       
#          Neg Pred Value : 0.8155       
#              Prevalence : 0.4920       
#          Detection Rate : 0.3926       
#    Detection Prevalence : 0.4615       
#       Balanced Accuracy : 0.8312       
#                                        
#        'Positive' Class : 0   


# Make predictions on the test data
predicted.classes <- model1 %>% 
  predict(tst, type = "class")
head(predicted.classes)

confusionMatrix(as.factor(predicted.classes), as.factor(tst$y_test))
# Confusion Matrix and Statistics
# 
#             Reference
# Prediction   0   1
# 0           122  62
# 1            91 141
# 
# Accuracy : 0.6322          
# 95% CI : (0.5839, 0.6787)
# No Information Rate : 0.512           
# P-Value [Acc > NIR] : 5.122e-07       
# 
# Kappa : 0.2665          
# 
# Mcnemar's Test P-Value : 0.02359         
#                                           
#             Sensitivity : 0.5728          
#             Specificity : 0.6946          
#          Pos Pred Value : 0.6630          
#          Neg Pred Value : 0.6078          
#              Prevalence : 0.5120          
#          Detection Rate : 0.2933          
#    Detection Prevalence : 0.4423          
#       Balanced Accuracy : 0.6337          
#                                           
#        'Positive' Class : 0  



## Pruning the Tree---------------------------------------------------------------------------------------------------------------------------

# Fit the model on the training set
set.seed(123)
model2 <- train(
  as.factor(y_train) ~., data = trn, method = "rpart",
  trControl = trainControl("cv", number = 10),
  tuneLength = 20)


# Plot model accuracy vs different values of complexity parameter)

ggplot(model2)+
  labs(title = "Model accuracy for different numbers of complexity parameter")+
  geom_point(color='indianred')+
  geom_line(color='darkblue')

# variable importance plot
vip(model2, num_features = 26,aesthetics = list(color = "dodgerblue", fill = "dodgerblue"))

#best cp that maximizes model accuracy
model2$bestTune                     # 0.01062918


# Final tree model
par(xpd = NA) 
plot(model2$finalModel)
text(model2$finalModel,  digits = 3)


# Decision rules in the model
model2$finalModel

# Make predictions on the train data
predicted.classes <- model2 %>% predict(trn)

confusionMatrix(factor(predicted.classes), factor(trn$y_train))
# 
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 231  52
# 1  76 265
# 
# Accuracy : 0.7949         
# 95% CI : (0.761, 0.8259)
# No Information Rate : 0.508          
# P-Value [Acc > NIR] : < 2e-16        
# 
# Kappa : 0.5891         
# 
# Mcnemar's Test P-Value : 0.04206        
#                                          
#             Sensitivity : 0.7524         
#             Specificity : 0.8360         
#          Pos Pred Value : 0.8163         
#          Neg Pred Value : 0.7771         
#              Prevalence : 0.4920         
#          Detection Rate : 0.3702         
#    Detection Prevalence : 0.4535         
#       Balanced Accuracy : 0.7942         
#                                          
#        'Positive' Class : 0              


# Make predictions on the test data
predicted.classes <- model2 %>% predict(tst)

confusionMatrix(factor(predicted.classes), factor(tst$y_test))
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 120  56
# 1  93 147
# 
# Accuracy : 0.6418         
# 95% CI : (0.5937, 0.688)
# No Information Rate : 0.512          
# P-Value [Acc > NIR] : 6.214e-08      
# 
# Kappa : 0.2863         
# 
# Mcnemar's Test P-Value : 0.003186       
#                                          
#             Sensitivity : 0.5634         
#             Specificity : 0.7241         
#          Pos Pred Value : 0.6818         
#          Neg Pred Value : 0.6125         
#              Prevalence : 0.5120         
#          Detection Rate : 0.2885         
#    Detection Prevalence : 0.4231         
#       Balanced Accuracy : 0.6438         
#                                          
#        'Positive' Class : 0            




## Bagging------------------------------------------------------------------------------------------------------------------------------------

set.seed(123)
pd_bag <- train(
  as.factor(y_train) ~ .,
  data = trn,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 10),
  nbagg = 300,  
  control = rpart.control(minsplit = 2, cp = 0)
)

# predictions on the train data
bag_pred = predict(pd_bag, trn)
confusionMatrix(as.factor(bag_pred), as.factor(trn$y_train))

# Reference
# Prediction   0   1
# 0 307   0
# 1   0 317
# 
# Accuracy : 1          
# 95% CI : (0.9941, 1)
# No Information Rate : 0.508      
# P-Value [Acc > NIR] : < 2.2e-16  
# 
# Kappa : 1          
# 
# Mcnemar's Test P-Value : NA         
#                                      
#             Sensitivity : 1.000      
#             Specificity : 1.000      
#          Pos Pred Value : 1.000      
#          Neg Pred Value : 1.000      
#              Prevalence : 0.492      
#          Detection Rate : 0.492      
#    Detection Prevalence : 0.492      
#       Balanced Accuracy : 1.000      
#                                      
#        'Positive' Class : 0 



# predictions on the test data
bag_pred = predict(pd_bag, tst)
confusionMatrix(as.factor(bag_pred), as.factor(tst$y_test))

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 122  34
# 1  91 169
# 
# Accuracy : 0.6995         
# 95% CI : (0.653, 0.7432)
# No Information Rate : 0.512          
# P-Value [Acc > NIR] : 5.949e-15      
# 
# Kappa : 0.4026         
# 
# Mcnemar's Test P-Value : 5.477e-07      
#                                          
#             Sensitivity : 0.5728         
#             Specificity : 0.8325         
#          Pos Pred Value : 0.7821         
#          Neg Pred Value : 0.6500         
#              Prevalence : 0.5120         
#          Detection Rate : 0.2933         
#    Detection Prevalence : 0.3750         
#       Balanced Accuracy : 0.7026         
#                                          
#        'Positive' Class : 0              



# Feature Importance
vip::vip(pd_bag, num_features = 26,aesthetics = list(color = "dodgerblue", fill = "dodgerblue"))



## Random Forests----------------------------------------------------------------------------------------------------------------------------



# number of features
n_features <- length(setdiff(names(trn), "y_train"))

# train a default random forest model (default for R in 500 trees)
pd_rf1 <- ranger(
  as.factor(y_train) ~ ., 
  data = trn,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123)


pd_rf1
# Type:                             Classification 
# Number of trees:                  500 
# Sample size:                      624 
# Number of independent variables:  26 
# Mtry:                             8 
# Target node size:                 1 
# Variable importance mode:         none 
# Splitrule:                        gini 
# OOB prediction error:             29.49 % 



# apply to train data
rf1_pred = predict(pd_rf1, trn)
rf1_pred$predictions

confusionMatrix(as.factor(rf1_pred$predictions), as.factor(y_train))    #500 trees



# apply to test data
rf1_pred = predict(pd_rf1, tst)
rf1_pred$predictions

confusionMatrix(as.factor(rf1_pred$predictions), as.factor(y_test))    #500 trees


# Following code is based on the book Hands-on Machine Learning with R (https://bradleyboehmke.github.io/HOML/)


## hyperparameter tuning:

# 1.The number of trees in the forest
# 2.The number of features to consider at any given split: mtry
# 3.The complexity of each tree
# 4.The sampling scheme
# 5.The splitting rule to use during tree construction


# Grid search

# hyperparameter grid
hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, .15, .25, .333, .4, .8)),
  min.node.size = c(1, 3, 5, 10), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(.5, .63, .8),                       
  error_perc = NA)


for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = as.factor(y_train) ~ ., 
    data            = trn, 
    num.trees       = n_features * 15,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 123,
    respect.unordered.factors = 'order',)
  
  hyper_grid$error_perc[i] <- fit$prediction.error
}

# the top 10 models arranged in terms of error
hyper_grid %>%
  arrange(error_perc) %>%
  head(10)

# impurity variable importance
rf_impurity <- ranger(
  formula = as.factor(y_train) ~ ., 
  data = trn, 
  num.trees = 2000,
  mtry = 6,
  min.node.size = 1,
  sample.fraction = .63,
  replace = FALSE,
  importance = "impurity",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 123)


#permutation variable importance
rf_permutation <- ranger(
  formula = as.factor(y_train) ~ ., 
  data = trn, 
  num.trees = 2000,
  mtry = 6,
  min.node.size = 1,
  sample.fraction = .63,
  replace = FALSE,
  importance = "permutation",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 123)

# variable importance plots
p1 <- vip::vip(rf_impurity, num_features = 26,aesthetics = list(color = "dodgerblue", fill = "dodgerblue"))
p2 <- vip::vip(rf_permutation, num_features = 26,aesthetics = list(color = "dodgerblue", fill = "dodgerblue"))
gridExtra::grid.arrange(p1, p2, nrow = 1)

# there appears to be enough evidence to suggest that three variables stand out as most influential:
# Maximum pitch
# Shimmer apq11
# Standard deviation


# final Random forest model after tuning
rf <- ranger(
  formula = as.factor(y_train) ~ ., 
  data = trn, 
  num.trees = 2000,
  mtry = 6,
  min.node.size = 1,
  sample.fraction = .63,
  replace = FALSE,
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 123)

# apply to train data
rf_pred = predict(rf, trn)

confusionMatrix(as.factor(rf_pred$predictions), as.factor(y_train)) 
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 307   0
# 1   0 317
# 
# Accuracy : 1          
# 95% CI : (0.9941, 1)
# No Information Rate : 0.508      
# P-Value [Acc > NIR] : < 2.2e-16  
# 
# Kappa : 1          
# 
# Mcnemar's Test P-Value : NA         
#                                      
#             Sensitivity : 1.000      
#             Specificity : 1.000      
#          Pos Pred Value : 1.000      
#          Neg Pred Value : 1.000      
#              Prevalence : 0.492      
#          Detection Rate : 0.492      
#    Detection Prevalence : 0.492      
#       Balanced Accuracy : 1.000      
#                                      
#        'Positive' Class : 0 



# apply to test data
rf_pred = predict(rf, tst)

confusionMatrix(as.factor(rf_pred$predictions), as.factor(y_test)) 
 
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 120  38
# 1  93 165
# 
# Accuracy : 0.6851          
# 95% CI : (0.6381, 0.7295)
# No Information Rate : 0.512           
# P-Value [Acc > NIR] : 6.046e-13       
# 
# Kappa : 0.3738          
# 
# Mcnemar's Test P-Value : 2.382e-06       
#                                           
#             Sensitivity : 0.5634          
#             Specificity : 0.8128          
#          Pos Pred Value : 0.7595          
#          Neg Pred Value : 0.6395          
#              Prevalence : 0.5120          
#          Detection Rate : 0.2885          
#    Detection Prevalence : 0.3798          
#       Balanced Accuracy : 0.6881          
#                                           
#        'Positive' Class : 0 







## Gradient Boosting-----------------------------------------------------------------------------------------------------------------------


# basic GBM model
set.seed(123)  # for reproducibility
pd_gbm1 <- gbm(
  formula = y_train ~ .,
  data = trn,
  n.trees = 300,
  shrinkage = 0.1,
  interaction.depth = 2,
  n.minobsinnode = 10,
  cv.folds = 10)


# minimum CV error
best <- which.min(pd_gbm1$cv.error)

pd_gbm1$cv.error[best]
# plot error curve
gbm.perf(pd_gbm1, method = "cv")  # 141


# grid search
hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  err = NA,
  trees = NA)


for(i in seq_len(nrow(hyper_grid))) {
  
  # fit gbm
  set.seed(123)  # for reproducibility
  m <- gbm(
    formula = y_train ~ .,
    data = trn,
    distribution = "bernoulli",
    n.trees = 5000, 
    shrinkage = hyper_grid$learning_rate[i], 
    interaction.depth = 3, 
    n.minobsinnode = 10,
    cv.folds = 10)
  
  # add SSE, trees, and training time to results
  hyper_grid$err[i]  <- min(m$cv.error)
  hyper_grid$trees[i] <- which.min(m$cv.error)
  
}

# results in terms of error
arrange(hyper_grid, err)


# search grid
hyper_grid <- expand.grid(
  n.trees = 236,
  shrinkage = 0.05,
  interaction.depth = c(2, 3, 5, 7),
  n.minobsinnode = c(1,5, 10, 15,20))


# create model fit function
model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(123)
  m <- gbm(
    formula = y_train ~ .,
    data = trn,
    distribution = "bernoulli",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    n.minobsinnode = n.minobsinnode,
    cv.folds = 10
  )
  # compute error
  min(m$cv.error) # 1.172112
}

library(purrr)
# grid with functional programming
hyper_grid$err <- purrr::pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)

# results
arrange(hyper_grid, err)
# n.trees shrinkage interaction.depth n.minobsinnode      err
# 1      236      0.05                 5              1 1.164556
# 2      236      0.05                 5             10 1.166100
# 3      236      0.05                 7              5 1.171885
# 4      236      0.05                 5              5 1.173122
# 5      236      0.05                 7             20 1.176816
# 6      236      0.05                 7             10 1.180866
# 7      236      0.05                 7              1 1.185611
# 8      236      0.05                 3              5 1.187984
# 9      236      0.05                 3             15 1.190253
# 10     236      0.05                 5             15 1.192309
# 11     236      0.05                 3             10 1.193693
# 12     236      0.05                 3              1 1.195463
# 13     236      0.05                 7             15 1.195725
# 14     236      0.05                 5             20 1.199648
# 15     236      0.05                 2              1 1.199782
# 16     236      0.05                 3             20 1.200050
# 17     236      0.05                 2              5 1.204298
# 18     236      0.05                 2             10 1.207139
# 19     236      0.05                 2             15 1.209721
# 20     236      0.05                 2             20 1.212970


# final GB model
set.seed(123)  # for reproducibility
final_gbm <- gbm(
  formula = y_train ~ .,
  data = trn,
  n.trees = 236,
  shrinkage = 0.05,
  interaction.depth = 5,
  n.minobsinnode = 1,
  cv.folds = 10)

# feature importance
vip(final_gbm, num_features = 26,aesthetics = list(color = "dodgerblue", fill = "dodgerblue"))

# apply to train data
gbm_pred = predict(final_gbm, trn)
f <- ifelse(gbm_pred>0.5,1,0)
confusionMatrix(as.factor(f), as.factor(trn$y_train))

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 304  46
# 1   3 271
# 
# Accuracy : 0.9215          
# 95% CI : (0.8975, 0.9413)
# No Information Rate : 0.508           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8433          
# 
# Mcnemar's Test P-Value : 1.973e-09       
#                                           
#             Sensitivity : 0.9902          
#             Specificity : 0.8549          
#          Pos Pred Value : 0.8686          
#          Neg Pred Value : 0.9891          
#              Prevalence : 0.4920          
#          Detection Rate : 0.4872          
#    Detection Prevalence : 0.5609          
#       Balanced Accuracy : 0.9226          
#                                           
#        'Positive' Class : 0               
#                                           
# 






# apply to test data
gbm_pred = predict(final_gbm, tst)

df <- data.frame(gbm_pred,tst$y_test)
library(plotROC)
rocplot <- ggplot(df, aes(m = gbm_pred, d = tst$y_test))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink")

f <- ifelse(gbm_pred>0.5,1,0)
confusionMatrix(as.factor(f), as.factor(tst$y_test))

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 150  75
# 1  63 128
# 
# Accuracy : 0.6683          
# 95% CI : (0.6207, 0.7134)
# No Information Rate : 0.512           
# P-Value [Acc > NIR] : 8.074e-11       
# 
# Kappa : 0.3352          
# 
# Mcnemar's Test P-Value : 0.3491          
#                                           
#             Sensitivity : 0.7042          
#             Specificity : 0.6305          
#          Pos Pred Value : 0.6667          
#          Neg Pred Value : 0.6702          
#              Prevalence : 0.5120          
#          Detection Rate : 0.3606          
#    Detection Prevalence : 0.5409          
#       Balanced Accuracy : 0.6674          
#                                           
#        'Positive' Class : 0               













## XGBoost--------------------------------------------------------------------------------------------------------------------------------

# basic model
set.seed(123)
base_xgb <- xgb.cv(
  data = X_train,
  label = y_train,
  nrounds = 6000,     # number of iterations
  objective = "binary:logistic",
  early_stopping_rounds = 50, 
  nfold = 10,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0),
  verbose = 0)  

min(base_xgb$evaluation_log$test_error_mean)  #  0.2919354


# hyperparameter grid
hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = 3, 
  min_child_weight = 3,
  subsample = 0.5, 
  colsample_bytree = 0.5,
  gamma = c(0, 1, 10, 100, 1000),
  lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  err = 0,                                     # error storing 
  trees = 0                                    #  number of trees storing
)

# grid search
for(i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m <- xgb.cv(
    data = X_train,
    label = y_train,
    nrounds = 6000,
    objective = "binary:logistic",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = hyper_grid$eta[i], 
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i], 
      lambda = hyper_grid$lambda[i], 
      alpha = hyper_grid$alpha[i]
    ) 
  )
  hyper_grid$err[i] <- min(m$evaluation_log$test_error_mean)
  hyper_grid$trees[i] <- m$best_iteration
}


# results
hyper_grid %>%
  arrange(err) %>%
  glimpse()


# best hyperparametrs after tuning
params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5)


# final model
xgb.fit.final <- xgboost(
  params = params,
  data = X_train,
  label = y_train,
  nrounds = 160,
  objective = "binary:logistic",
  verbose = 0)

# feature importance
vip(xgb.fit.final,aesthetics = list(color = "dodgerblue", fill = "dodgerblue"),num_features = 26) 



# apply to train data
xgpred = predict(xgb.fit.final,X_train)

df <- data.frame(xgpred,trn$y_train)

f <- ifelse(xgpred>0.5,1,0)
confusionMatrix(as.factor(f), as.factor(trn$y_train))

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 222  45
# 1  85 272
# 
# Accuracy : 0.7917          
# 95% CI : (0.7577, 0.8229)
# No Information Rate : 0.508           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.5824          
# 
# Mcnemar's Test P-Value : 0.000625        
#                                           
#             Sensitivity : 0.7231          
#             Specificity : 0.8580          
#          Pos Pred Value : 0.8315          
#          Neg Pred Value : 0.7619          
#              Prevalence : 0.4920          
#          Detection Rate : 0.3558          
#    Detection Prevalence : 0.4279          
#       Balanced Accuracy : 0.7906          
#                                           
#        'Positive' Class : 0               


# apply to test data
xgpred = predict(xgb.fit.final,X_test)

df <- data.frame(xgpred,tst$y_test)
library(plotROC)
rocplot <- ggplot(df, aes(m = xgpred, d = tst$y_test))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink")

f <- ifelse(xgpred>0.5,1,0)
confusionMatrix(as.factor(f), as.factor(tst$y_test))


# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 118  52
# 1  95 151
# 
# Accuracy : 0.6466          
# 95% CI : (0.5986, 0.6926)
# No Information Rate : 0.512           
# P-Value [Acc > NIR] : 2.037e-08       
# 
# Kappa : 0.2964          
# 
# Mcnemar's Test P-Value : 0.000532        
#                                           
#             Sensitivity : 0.5540          
#             Specificity : 0.7438          
#          Pos Pred Value : 0.6941          
#          Neg Pred Value : 0.6138          
#              Prevalence : 0.5120          
#          Detection Rate : 0.2837          
#    Detection Prevalence : 0.4087          
#       Balanced Accuracy : 0.6489          
#                                           
#        'Positive' Class : 0       









## AdaBoost---------------------------------------------------------------------------------------------------------------------------------


cvcontrol <- trainControl(method="repeatedcv", number = 10,
                          allowParallel=TRUE)

train.ada <- train(as.factor(y_train) ~ ., 
                   data=trn,
                   method="ada",
                   verbose=F,
                   trControl=cvcontrol)
train.ada

#obtaining class predictions for train data
ada.classTrain <-  predict(train.ada, 
                           newdata = trn,
                           type="raw")


#computing confusion matrix
confusionMatrix(as.factor(trn$y_train),as.factor(ada.classTrain))

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 237  70
# 1  30 287
# 
# Accuracy : 0.8397          
# 95% CI : (0.8086, 0.8677)
# No Information Rate : 0.5721          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.6787          
# 
# Mcnemar's Test P-Value : 9.619e-05       
#                                           
#             Sensitivity : 0.8876          
#             Specificity : 0.8039          
#          Pos Pred Value : 0.7720          
#          Neg Pred Value : 0.9054          
#              Prevalence : 0.4279          
#          Detection Rate : 0.3798          
#    Detection Prevalence : 0.4920          
#       Balanced Accuracy : 0.8458          
#                                           
#        'Positive' Class : 0 



#Obtaining predicted probabilites for Test data
ada.probs=predict(train.ada,
                  newdata=tst,
                  type="prob")


ada.classTest <-  predict(train.ada, 
                          newdata = tst,
                          type="raw")

#computing confusion matrix
confusionMatrix(as.factor(tst$y_test),as.factor(ada.classTest))

# 
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1
# 0 120  93
# 1  50 153
# 
# Accuracy : 0.6562          
# 95% CI : (0.6084, 0.7018)
# No Information Rate : 0.5913          
# P-Value [Acc > NIR] : 0.0038497       
# 
# Kappa : 0.3155          
# 
# Mcnemar's Test P-Value : 0.0004444       
#                                           
#             Sensitivity : 0.7059          
#             Specificity : 0.6220          
#          Pos Pred Value : 0.5634          
#          Neg Pred Value : 0.7537          
#              Prevalence : 0.4087          
#          Detection Rate : 0.2885          
#    Detection Prevalence : 0.5120          
#       Balanced Accuracy : 0.6639          
#                                           
#        'Positive' Class : 0 


df <- data.frame(ada.probs[,"1"] ,tst$y_test)
library(plotROC)
rocplot <- ggplot(df, aes(m = ada.probs[,"1"] , d = tst$y_test))+ geom_roc(n.cuts=20,labels=FALSE)
rocplot + style_roc(theme = theme_grey) + geom_rocci(fill="pink")

