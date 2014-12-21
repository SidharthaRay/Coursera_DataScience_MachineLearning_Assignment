---
title: "Practical Machine Learning"
subtitle: "Course Project"
author: "Sidhartha Ray"
date: "December 19, 2014"
output: html_document
---


Online version available here: http://rpubs.com/ldamewood/28262

### Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement---a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Load Data
In this section, load the data and the 20 cases that will be submitted to coursera.

```r
rm(list = ls())
if (!file.exists("pml-training.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
}
submit <- read.csv("pml-testing.csv", sep = ",", na.strings = c("", "NA"))
data <- read.csv("pml-training.csv", sep = ",", na.strings = c("", "NA"))
```

### Cleanup the data
Here, I remove columns full of NAs and remove features that are not in the submit set. The features containing NAs are the variance, mean and stddev within each window for each feature. Since the `submit` dataset has no time-dependence, these values are useless and can be disregarded. I also remove the first 7 features since they are related to the time-series or are not numeric.


```r
# Remove columns full of NAs.
features <- names(submit[,colSums(is.na(submit)) == 0])[8:59]
# Only use features used in submit cases.
data <- data[,c(features,"classe")]
submit <- submit[,c(features,"problem_id")]
```

### Bootstrap
Next, I withhold 25% of the dataset for testing after the final model is constructed.

```r
set.seed(916)
inTrain = createDataPartition(data$classe, p = 0.75, list = F)
training = data[inTrain,]
testing = data[-inTrain,]
```

### Feature Selection
Some features may be highly correlated. The PCA method mixes the final features into components that are difficult to interpret; instead, I drop features with high correlation (>90%).

```r
outcome = which(names(training) == "classe")
highCorrCols = findCorrelation(abs(cor(training[,-outcome])),0.90)
highCorrFeatures = names(training)[highCorrCols]
training = training[,-highCorrCols]
outcome = which(names(training) == "classe")
```

The features with high correlation are accel_belt_z, roll_belt, accel_belt_y, accel_belt_x, gyros_arm_y, gyros_forearm_z, and gyros_dumbbell_x.

### Feature Importance
The random forest method reduces overfitting and is good for nonlinear features. First, to see if the data is nonlinear, I use the random forest to discover the most important features. The feature plot for the 4 most important features is shown.

```r
fsRF = randomForest(training[,-outcome], training[,outcome], importance = T)
rfImp = data.frame(fsRF$importance)
impFeatures = order(-rfImp$MeanDecreaseGini)
inImp = createDataPartition(data$classe, p = 0.05, list = F)
featurePlot(training[inImp,impFeatures[1:4]],training$classe[inImp], plot = "pairs")
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png) 

The most important features are:

* `pitch_belt`
* `yaw_belt`
* `total_accel_belt`
* `gyros_belt_x`

### Training
Train using the random forest and k-nearest neighbors for comparison.

```r
ctrlKNN = trainControl(method = "adaptive_cv")
modelKNN = train(classe ~ ., training, method = "knn", trControl = ctrlKNN)
ctrlRF = trainControl(method = "oob")
modelRF = train(classe ~ ., training, method = "rf", ntree = 200, trControl = ctrlRF)
resultsKNN = data.frame(modelKNN$results)
resultsRF = data.frame(modelRF$results)
```

### Testing Out-of-sample error
The random forest will give a larger accuracy compared to k-nearest neighbors. Here, I give the confusion matrix between the KNN and RF models to see how much they agree on the test set, then I compare each model using the test set outcomes.

```r
fitKNN = predict(modelKNN, testing)
fitRF = predict(modelRF, testing)
```
#### KNN vs. RF

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1360    8   12   16    7
##          B   37  834   27   24   25
##          C   14   31  780   22    9
##          D    9    3   56  721   14
##          E   14   33   22   36  790
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9146          
##                  95% CI : (0.9064, 0.9222)
##     No Information Rate : 0.2924          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8918          
##  Mcnemar's Test P-Value : 3.105e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9484   0.9175   0.8696   0.8803   0.9349
## Specificity            0.9876   0.9717   0.9810   0.9799   0.9741
## Pos Pred Value         0.9694   0.8807   0.9112   0.8979   0.8827
## Neg Pred Value         0.9789   0.9810   0.9711   0.9761   0.9863
## Prevalence             0.2924   0.1854   0.1829   0.1670   0.1723
## Detection Rate         0.2773   0.1701   0.1591   0.1470   0.1611
## Detection Prevalence   0.2861   0.1931   0.1746   0.1637   0.1825
## Balanced Accuracy      0.9680   0.9446   0.9253   0.9301   0.9545
```
#### KNN vs. test set

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1354   44   13    9   14
##          B    6  835   32    3   33
##          C   12   25  779   59   22
##          D   16   24   20  723   36
##          E    7   21   11   10  796
## 
## Overall Statistics
##                                           
##                Accuracy : 0.915           
##                  95% CI : (0.9068, 0.9226)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8924          
##  Mcnemar's Test P-Value : 3.911e-15       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9706   0.8799   0.9111   0.8993   0.8835
## Specificity            0.9772   0.9813   0.9709   0.9766   0.9878
## Pos Pred Value         0.9442   0.9186   0.8685   0.8828   0.9420
## Neg Pred Value         0.9882   0.9715   0.9810   0.9802   0.9741
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2761   0.1703   0.1588   0.1474   0.1623
## Detection Prevalence   0.2924   0.1854   0.1829   0.1670   0.1723
## Balanced Accuracy      0.9739   0.9306   0.9410   0.9379   0.9356
```
#### RF vs. test set

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    8    0    0    0
##          B    0  940    7    0    0
##          C    0    1  845   10    0
##          D    0    0    3  794    6
##          E    0    0    0    0  895
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9929         
##                  95% CI : (0.9901, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.991          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9905   0.9883   0.9876   0.9933
## Specificity            0.9977   0.9982   0.9973   0.9978   1.0000
## Pos Pred Value         0.9943   0.9926   0.9871   0.9888   1.0000
## Neg Pred Value         1.0000   0.9977   0.9975   0.9976   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1917   0.1723   0.1619   0.1825
## Detection Prevalence   0.2861   0.1931   0.1746   0.1637   0.1825
## Balanced Accuracy      0.9989   0.9944   0.9928   0.9927   0.9967
```
The random forest fit is clearly more accurate than the k-nearest neighbors method with 99% accuracy.

### Submit
Finally, I use the random forest model to preduct on the 20 cases submitted to coursera.

```
##  problem.id answers
##           1       B
##           2       A
##           3       B
##           4       A
##           5       A
##           6       E
##           7       D
##           8       B
##           9       A
##          10       A
##          11       B
##          12       C
##          13       B
##          14       A
##          15       E
##          16       E
##          17       A
##          18       B
##          19       B
##          20       B
```
