### The R code for Random Forest
library(dplyr)
library(plotly)
library(GGally)
library("ggplot2")
library(leaps)
library(MASS)
library(glmnet)
library(pls)
library(randomForest)
library(adabag)
library(caret)
library(ada)
library(corrplot)
library(nnet)
library(caTools)
library(ROCR)
library(pROC)
library(randomForest)
library('smotefamily')
library("ROSE")
### Data Preparation 
## Read Data 
rm(list = ls())
water_potability <- read.csv("~/Downloads/water_potability.csv")
## Split to training and testing subset 
set.seed(123)
clean_data<- na.omit(water_potability)
clean_data$Potability=as.factor(clean_data$Potability);
n<-dim(clean_data)[1]
flag <- sort(sample(n,floor(0.2*n), replace = FALSE))
datatrain <- clean_data[-flag,]
datatest <- clean_data[flag,]

train_std_deviation= apply(datatrain[,-10],2,sd)
train_mean= apply(datatrain[,-10],2,mean)

scaled_train =  (datatrain[,-10] - train_mean) / train_std_deviation
scaled_test = (datatest[,-10] - train_mean) / train_std_deviation
scaled_train$Potability=datatrain[,10]
scaled_test$Potability=datatest[,10]
## Extra the true response value for training and testing data
y1  <- as.factor(datatrain$Potability);
y2   <- as.factor(datatest$Potability);



## Data Exploratory Analysis (EDA)
ggpairs(clean_data, columns = 1:9)#+theme(strip.text = element_text(size = 1))

#ggplotly(p1)


#correlation Matrix
M=cor(clean_data[,-10])
corrplot(M)
# PCA representation

library(ggfortify)
# PCA visualization
df <- datatrain
df$Potability = as.numeric(df$Potability)

#df$Patability=as.numeric(datatrain$Potability)
pca_im <- prcomp(df, scale. = TRUE)

autoplot(pca_im)

a<-autoplot(pca_im, data = datatrain, colour = 'Potability')
a

library(ggplot2)
# Bar chart
ggplot(datatrain, aes(x=Potability,fill=Potability))+geom_bar(stat="count", width=0.7)+
geom_text(stat='count', aes(label=..count..), vjust=1)+
labs(title = " number of potabale and non-potable samples  ",x="Potability", y="Number of samples")+theme_bw()

# Use semi-transparent fill
ph<-ggplot(datatrain, aes(x=ph, fill=Potability, color=Potability)) +
  geom_histogram(position="identity", alpha=0.5)+labs(y= "number of Samples", x = "Ph")

ph

Hardness<-ggplot(datatrain, aes(x=Hardness, fill=Potability, color=Potability)) +
  geom_histogram(position="identity", alpha=0.5)+labs(y= "number of Samples", x = "Hardness")

Hardness

Org<-ggplot(datatrain, aes(x=Organic_carbon, fill=Potability, color=Potability)) +
  geom_histogram(position="identity", alpha=0.5)+labs(y= "number of Samples", x = "Organic Carbon")

Org


S<-ggplot(datatrain, aes(x=Turbidity, fill=Potability, color=Potability)) +
  geom_histogram(position="identity", alpha=0.5)+labs(y= "number of Samples", x = "Solids")

S
###################################################
# Using Balanced data
datatrain<- ROSE(Potability~ ., data = datatrain,seed = 1)$data
ggplot(datatrain, aes(x=Potability,fill=Potability))+geom_bar(stat="count", width=0.7)+
  geom_text(stat='count', aes(label=..count..), vjust=1)+
  labs(title = " number of potabale and non-potable samples  ",x="Potability", y="Number of samples")+theme_bw()
################################################

# Cross Validation for Random Forest
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(1)
mtry <- sqrt(ncol(datatrain))
#rf1 <- randomForest(y1 ~., data=datatrain[,-10], 
#                    ntree= 600, 
 #                   mtry=mtry, nodesize =2, importance=TRUE)
rf1 <- train(Potability~., data=datatrain, method="rf", metric="Accuracy", tuneLength=15, trControl=control)
print(rf1)
plot(rf1)

## Check Important variables
importance(rf1)
rf <- randomForest(y1~., data=datatrain[,-10], mtry=6)
## There are two types of importance measure 
##  (1=mean decrease in accuracy, 
##   2= mean decrease in node impurity)
importance(rf, type=2)
varImpPlot(rf)

## The plots show that V52, V53, V7, V55 are among the most 
##     important features when predicting V58. 

## Prediction on the testing data set
rf.pred = predict(rf, datatest[,-10], type='class')
t1= table(rf.pred, y2)
sum(diag(table(rf.pred, y2)))/sum(table(rf.pred,y2)) 

rc1= confusionMatrix(t1)
precision <- rc1$byClass['Pos Pred Value']
precision

##In practice, You can fine-tune parameters in Random Forest such as 
#ntree = number of tress to grow, and the default is 500. 
#mtry = number of variables randomly sampled as candidates at each split. 
#          The default is sqrt(p) for classfication and p/3 for regression
#nodesize = minimum size of terminal nodes. 
#           The default value is 1 for classification and 5 for regression
plot(rf)

### ababoost
datatrain$Potability <- as.factor(datatrain$Potability)
model_adaboost <- boosting(Potability~., data=datatrain,boos=TRUE)
summary(model_adaboost)
pred_test1 = predict(model_adaboost, datatest)
pred_test1$error
sum(diag(pred_test$confusion))/sum(pred_test$confusion)

t6= table(as.factor(pred_test1),  datatest$Potability) 
rc6= confusionMatrix(t6)
precision <- rc6$byClass['Pos Pred Value']
precision



### KNN, Linear , LDA, logistic regression
# Using KNN
library(class);
xnew <- scaled_train[,1:9];
training_err=c()
for (kk in c(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31)){
  
  ypred2.train <- knn(scaled_train[,1:9], xnew, scaled_train[,10], k=kk);
  training_err= c(training_err,mean( ypred2.train != scaled_train[,10]))
}
# KNN training error for each K value
training_err
plot(c(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31), 
     training_err,main ="K-nearest Vs Training Error", type = "b", pch = 19,
     col = "red", xlab = "K-nearest", ylab = "Training Error")
### 3. Testing Error

## Testing error of KNN, and you can change the k values.
xnew2 <- scaled_test[,1:9]; ## xnew2 is the X variables of the "testing" data

ypred2.test <- knn(scaled_train[,1:9], xnew2, scaled_train[,10], k=11);
test_err= mean( ypred2.test!= scaled_test[,10])
t1= table(ypred2.test,  datatest$Potability)
rc1= confusionMatrix(t1)
precision <- rc1$byClass['Pos Pred Value']
# Test error for different k values 
test_err
precision


TrainErr <- NULL;
TestErr  <- NULL; 
#selecting only the variables that are useful in predicting mpg
mod1 <- lda(datatrain[,1:9], datatrain[,10]); 

## training error 
## we provide a detailed code here 
pred1 <- predict(mod1,datatrain[,1:9])$class; 
lda_TrainErr <- mean( pred1 != datatrain$Potability); 
lda_TrainErr; 
## testing error 
pred1test <- predict(mod1,datatest[,1:9])$class; 
lda_testerr=mean(pred1test != datatest$Potability) 
lda_testerr
t2= table(pred1test,  datatest$Potability) 
rc2= confusionMatrix(t2)
precision <- rc2$byClass['Pos Pred Value']
precision
## Method 2: QDA
mod2 <- qda(datatrain[,1:9], datatrain[,10])
## Training Error 
pred2 <- predict(mod2,datatrain[,1:9])$class
qda_trainerr=mean( pred2!= datatrain$Potability)
qda_trainerr

##  Testing Error 
qda_testerr=mean( predict(mod2,datatest[,1:9])$class != datatest$Potability)
qda_testerr

##Logistic Regression
# Training model
logistic_model <- glm(Potability ~ ., 
                      data = scaled_train, 
                      family = "binomial")


# Summary
summary(logistic_model)

# Predict test data based on model
predtrain <- predict(logistic_model, 
                     scaled_train[,1:9], type = "response")


# figuring out the best probability treshhold to categorize the data  
prediction(predtrain, scaled_train$Potability) %>%
  performance(measure = "tpr", x.measure = "fpr") -> result

plotdata <- data.frame(x = result@x.values[[1]],
                       y = result@y.values[[1]], 
                       p = result@alpha.values[[1]])

p <- ggplot(data = plotdata) +
  geom_path(aes(x = x, y = y)) + 
  xlab(result@x.name) +
  ylab(result@y.name) +
  theme_bw()

dist_vec <- plotdata$x^2 + (1 - plotdata$y)^2
opt_pos <- which.min(dist_vec)

p + 
  geom_point(data = plotdata[opt_pos, ], 
             aes(x = x, y = y), col = "red") +
  annotate("text", 
           x = plotdata[opt_pos, ]$x + 0.1,
           y = plotdata[opt_pos, ]$y,
           label = paste("p =", round(plotdata[opt_pos, ]$p, 3)))
# Changing probabilities
predtr<- ifelse(predtrain >0.411, 1, 0)
# training error
logit_trainerr=mean(predtr != datatrain$Potability)
logit_trainerr
# Evaluating model accuracy
# using confusion matrix


table(datatrain$Potability, predtr)

predtest<- predict(logistic_model, 
                   datatest[,1:9], type = "response")

# Changing probabilities
predts<- ifelse(predtest >0.411, 1, 0)
# training error
logit_testerr= mean(predts!= scaled_test$Potability)
logit_testerr
table(predts,  scaled_test$Potability) 

t3= table(predts,  datatest$Potability) 
rc3= confusionMatrix(t3)
precision <- rc3$byClass['Pos Pred Value']
precision
# Naive bayes

library(e1071)
mod3 <- naiveBayes( datatrain[,1:9], datatrain[,10])
#mod3 <- naive_bayes(Potability~ ., data = datatrain,type = 'prob' ) 
## Training Error
prednb3 <- predict(mod3, datatrain)
(tab2 <- table(prednb3 , datatrain$Potability))
NBtrainerr= mean( prednb3 != datatrain$Potability)
NBtrainerr
##  0.2765152 for miss.class.train.error of Naive Bayes
## Testing Error 
predt3 <- predict(mod3, datatest)
NBtesterr= mean( predict(mod3,datatest) != datatest$Potability)
NBtesterr
t4= table(predt3,  datatest$Potability) 
rc4= confusionMatrix(t4)
precision <- rc4$byClass['Pos Pred Value']
precision
# SVM 
library(kernlab)
## Model #4: Gaussian-kernel SVM 
fit4 <- ksvm(Potability ~ ., data = datatrain,C=1, kernel="rbfdot")
y_pred= predict(fit4, datatest,type ="response")
ytest=datatest$Potability
table(y_pred,ytest)
sum(diag(table(y_pred,ytest)))/sum(table(y_pred,ytest))
##   This should be the same model #1 from the "e1701" package 



t5= table(y_pred,  datatest$Potability) 
rc5= confusionMatrix(t5)
precision <- rc5$byClass['Pos Pred Value']
precision

#pca_res <- prcomp(clean_data[,-10], scale. = TRUE)
#autoplot(pca_res, data=clean_data[,-10], colour=clean_data[,1])

library(MASS)
full.model <- glm(Potability ~., data = datatrain, family = binomial)
coef(full.model)
step.model <- full.model %>% stepAIC(trace = FALSE)
coef(step.model)



# Make predictions
probabilities <- full.model %>% predict(datatest[,-10], type = "response")
predicted.classes <- ifelse(probabilities > 0.411, 1, 0)

# Balancing the data using Rose Package 
t5= table(predicted.classes,  datatest$Potability) 
rc5= confusionMatrix(t5)
precision <- rc5$byClass['Pos Pred Value']
precision
# Prediction accuracy
mean(predicted.classes == scaled_test$Potability)
# Make predictions
probabilities <- predict(step.model, datatest[,-10], type = "response")
predicted.classes <- ifelse(probabilities > 0.411, 1, 0)
# Prediction accuracy
t5= table(predicted.classes,  datatest$Potability) 
rc5= confusionMatrix(t5)
precision <- rc5$byClass['Pos Pred Value']
precision
mean(predicted.classes == scaled_test$Potability)



