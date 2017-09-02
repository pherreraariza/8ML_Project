# set the working directory and upload libraries
setwd("//Users/pherreraariza/Documents/Coursera/8_Machine_Learning/8ML_Project")
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(nnet)

# set the URL for the download
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
training <- read.csv(url(UrlTrain))
testing  <- read.csv(url(UrlTest))

# create a partition with the training dataset 
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]

# delete near zero variables
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]

# remove variables that are mostly NA
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]

# remove identification only variables (columns 1 to 5)
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]

# correlation matrix
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "circle", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))

# model fit RF
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf", trControl=controlRF)
modFitRandForest$finalModel

# prediction on Test dataset
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)
confMatRandForest

# plot matrix results
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))

# model fit TREE
set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)

# prediction on Test dataset
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree

# plot matrix results
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))

# model fit GBM
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel

# prediction on Test dataset
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM

# plot matrix results
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))

# model fit NN
set.seed(12345)
controlNN <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitNN  <- train(classe ~ ., data=TrainSet, method = "pcaNNet", 
                    trControl = controlNN, verbose = FALSE)
modFitNN$finalModel

# prediction on Test dataset
predictNN <- predict(modFitNN, newdata=TestSet)
confMatNN <- confusionMatrix(predictNN, TestSet$classe)
confMatNN

# plot matrix results
plot(confMatNN$table, col = confMatNN$byClass, 
     main = paste("Neural Net - Accuracy =", round(confMatNN$overall['Accuracy'], 4)))


# model fit SVM
set.seed(12345)
controlSVM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitSVM  <- train(classe ~ ., data=TrainSet, method = "svmRadial", 
                   trControl = controlSVM, verbose = FALSE)
modFitSVM$finalModel

# prediction on Test dataset
predictSVM <- predict(modFitSVM, newdata=TestSet)
confMatSVM <- confusionMatrix(predictSVM, TestSet$classe)
confMatSVM

# plot matrix results
plot(confMatSVM$table, col = confMatSVM$byClass, 
     main = paste("SVM- Accuracy =", round(confMatSVM$overall['Accuracy'], 4)))
