Weight Lifting Exercise data analysis
========================================================
  
  
# Summary 
In this paper Weight Lifting Exercise data is analyzed to build a model which allows to make predictions on which class, among 5 possible values from A to E, data belongs to.

After an initial data preprocessing, a total of 7 different classification models (Random Forest, Knn, C5, GBM, Linear, Quadratic and Flexible Discrmininat Analysis techniques) will be built on a training set representing the 70% of initial data set.

The performance of these models, say their accuracy, will be evaluated on the Cross Validation set, represented by the remaining 30% of the data.

The several models will be also compared not only in terms of their accuracy on the cross validation data but also in terms of training time (i.e. time needed by each model to train the data set). 

As a result, it will be shown that Random Forest has a superb performance with a 99,3% accuracy on the cross validation set. However a model based on the C5 algorithm, while reaching the same exact accuracy of Random Forest, takes 1/3 of elapse time to train the same data set.

In other words, RF and c5 are indistinguishable models from an accuracy point of view but c5 is a much faster algorithm than Random Forest. At least on this data set.  

In particular cases where computing resources or timing is an issue, knn may become a excellent alternative because while not reaching an accuracy as high as c5, its training time is much lower than both c5 and Random Forest


# Preliminary set-up

## Loading and preprocessing
Let's first load the data from the training file and make the introductory manipulations.  
In order we will:  
1. read the data  
2. replace non numeric values with NA's  
3. exclude the first 7 columns which have nothing to do with the output  
4. exclude those variables with more than 95% of NA values  
 
 

```r
library(caret)
initialTrainingDataset <- read.csv("pml-training.csv", header = TRUE)
```





```r
suppressWarnings(for (i in 1:(ncol(initialTrainingDataset) - 1)) initialTrainingDataset[, 
    i] <- as.numeric(as.character(initialTrainingDataset[, i])))
```




```r
# exclude the first 7 columns
PartiallyCleanedTrainingData <- initialTrainingDataset[, -c(1:7)]

# exclude all columns with more than 95% of NAs values
FinalTrainingData <- PartiallyCleanedTrainingData[, -which(colSums(is.na(PartiallyCleanedTrainingData))/nrow(PartiallyCleanedTrainingData) > 
    0.95)]
```





## Split in training and test sets
Training set will be extracted through the function  "createDataPartition" in the caret package which will make a "stratified" extraction (i.e. maintaining the output class balance as in the original data set).
Since there is a large amount of data available, we can build the models using up to 70% of data while the remaining 30% will be used to evaluate accuracies.


```r
inTrain <- createDataPartition(y = FinalTrainingData$classe, p = 0.7, list = FALSE)
train <- FinalTrainingData[inTrain, ]
crossValidation <- FinalTrainingData[-inTrain, ]
dim(train)
```

```
## [1] 13737    53
```

```r
dim(crossValidation)
```

```
## [1] 5885   53
```



## Class balance check
In classifications problems it is important to check weather classes are balanced. 


```r
table(train$classe)
```

```
## 
##    A    B    C    D    E 
## 3906 2658 2396 2252 2525
```

In this case, while class A is much more populated than the other classes, the imbalance is not to worry about. 

## Correlation check

```r
M <- abs(cor(train[, -53]))
diag(M) <- 0
which(M > 0.8, arr.ind = T)
```

```
##                  row col
## yaw_belt           3   1
## total_accel_belt   4   1
## accel_belt_y       9   1
## accel_belt_z      10   1
## accel_belt_x       8   2
## magnet_belt_x     11   2
## roll_belt          1   3
## roll_belt          1   4
## accel_belt_y       9   4
## accel_belt_z      10   4
## pitch_belt         2   8
## magnet_belt_x     11   8
## roll_belt          1   9
## total_accel_belt   4   9
## accel_belt_z      10   9
## roll_belt          1  10
## total_accel_belt   4  10
## accel_belt_y       9  10
## pitch_belt         2  11
## accel_belt_x       8  11
## gyros_arm_y       19  18
## gyros_arm_x       18  19
## magnet_arm_x      24  21
## accel_arm_x       21  24
## magnet_arm_z      26  25
## magnet_arm_y      25  26
## accel_dumbbell_x  34  28
## accel_dumbbell_z  36  29
## pitch_dumbbell    28  34
## yaw_dumbbell      29  36
```



## Models building
In the following several classification models will be applied on training data:  
1. RF: Random forest  
2. LDA: Linear Discriminant Analysis  
3. QDA: Quadratic Discriminant Analysis  
4. FDA: Flexible Discriminant Analysis  
5. GBM: Gradient boosting  
6. KNN: k-nearest neighbors  
7. C5: evolution of C4.5  

Center and scale preprocessing will be applied exclusively to LDA, QDA, FDA and KNN since the other models are not sensitive to such preprocessing (refer to "applied predictive modeling", by Max Kuhn and Kjell Johnson 2013, pg. 550).

Through the function proc.time(), the processing time of each method will be also evaluated and stored for subsequent comparative analysis. 

### RANDOM FOREST


```r
set.seed(100)
ptm <- proc.time()
rf <- train(classe ~ ., data = train, method = "rf", prox = TRUE, trControl = trainControl(method = "cv", 
    number = 10))
rf_time <- proc.time() - ptm
```




### LDA

```r
set.seed(100)
ptm <- proc.time()
lda <- train(classe ~ ., data = train, method = "lda", preProc = c("center", 
    "scale"), trControl = trainControl(method = "cv", number = 10))
LDA_time <- proc.time() - ptm
```


### QDA

```r

set.seed(100)
ptm <- proc.time()
qda <- train(classe ~ ., data = train, method = "qda", preProc = c("center", 
    "scale"), trControl = trainControl(method = "cv", number = 10))
QDA_time <- proc.time() - ptm
```





### FDA

```r

set.seed(100)
ptm <- proc.time()
fda <- train(classe ~ ., data = train, method = "fda", preProc = c("center", 
    "scale"), trControl = trainControl(method = "cv", number = 10))
FDA_time <- proc.time() - ptm
```







### GBM

```r
set.seed(100)
ptm <- proc.time()
GBM <- train(classe ~ ., method = "gbm", data = train, trControl = trainControl(method = "cv", 
    number = 10))
GBM_time <- proc.time() - ptm
```


### KNN

```r

set.seed(100)
ptm <- proc.time()
knn <- train(classe ~ ., data = train, method = "knn", preProc = c("center", 
    "scale"), trControl = trainControl(method = "cv", number = 10))
knn_time <- proc.time() - ptm
```


### C5

```r
set.seed(100)
ptm <- proc.time()
c5 <- train(classe ~ ., data = train, trControl = trainControl(method = "cv", 
    number = 10))
c5_time <- proc.time() - ptm
```


### BAG




### GAM

```r
# gam <- train(classe ~ ., data = train, method = 'gam', trControl =
# trainControl##(method = 'cv',number = 10))
```



# Accuracy evaluation on the cross validation set and best model selection

## Confusion matrix

Confusion matrix for each model will now be calculated on the cross validation set.


```r
cm_rf <- confusionMatrix(crossValidation$classe, predict(rf, crossValidation))
cm_knn <- confusionMatrix(crossValidation$classe, predict(knn, crossValidation))
cm_c5 <- confusionMatrix(crossValidation$classe, predict(c5, crossValidation))
cm_lda <- confusionMatrix(crossValidation$classe, predict(lda, crossValidation))
cm_qda <- confusionMatrix(crossValidation$classe, predict(qda, crossValidation))
cm_fda <- confusionMatrix(crossValidation$classe, predict(fda, crossValidation))
cm_GBM <- confusionMatrix(crossValidation$classe, predict(GBM, crossValidation))
```


As you can see in the respective confusion matrix, C5 and Random forest are equivalent classification models on this data set.  



```r
cm_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    3 1136    0    0    0
##          C    0   15 1009    2    0
##          D    0    0   19  945    0
##          E    0    0    0    5 1077
## 
## Overall Statistics
##                                        
##                Accuracy : 0.992        
##                  95% CI : (0.99, 0.994)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.99         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.986    0.982    0.993    1.000
## Specificity             1.000    0.999    0.996    0.996    0.999
## Pos Pred Value          0.999    0.997    0.983    0.980    0.995
## Neg Pred Value          0.999    0.997    0.996    0.999    1.000
## Prevalence              0.285    0.196    0.175    0.162    0.183
## Detection Rate          0.284    0.193    0.171    0.161    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.993    0.989    0.994    0.999
```

```r
cm_c5
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    3 1136    0    0    0
##          C    0   15 1009    2    0
##          D    0    0   19  945    0
##          E    0    0    0    5 1077
## 
## Overall Statistics
##                                        
##                Accuracy : 0.992        
##                  95% CI : (0.99, 0.994)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.99         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.986    0.982    0.993    1.000
## Specificity             1.000    0.999    0.996    0.996    0.999
## Pos Pred Value          0.999    0.997    0.983    0.980    0.995
## Neg Pred Value          0.999    0.997    0.996    0.999    1.000
## Prevalence              0.285    0.196    0.175    0.162    0.183
## Detection Rate          0.284    0.193    0.171    0.161    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.993    0.989    0.994    0.999
```

```r
cm_knn
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1652    8    7    5    2
##          B   27 1076   31    1    4
##          C    2   15  991   15    3
##          D    2    1   39  916    6
##          E    1   12   11    5 1053
## 
## Overall Statistics
##                                         
##                Accuracy : 0.967         
##                  95% CI : (0.962, 0.971)
##     No Information Rate : 0.286         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.958         
##  Mcnemar's Test P-Value : 1.99e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.981    0.968    0.918    0.972    0.986
## Specificity             0.995    0.987    0.993    0.990    0.994
## Pos Pred Value          0.987    0.945    0.966    0.950    0.973
## Neg Pred Value          0.992    0.992    0.982    0.995    0.997
## Prevalence              0.286    0.189    0.183    0.160    0.181
## Detection Rate          0.281    0.183    0.168    0.156    0.179
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.988    0.977    0.956    0.981    0.990
```

```r
cm_lda
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1381   37  135  118    3
##          B  186  709  139   46   59
##          C   98   88  690  131   19
##          D   47   48  113  714   42
##          E   42  176  105   93  666
## 
## Overall Statistics
##                                         
##                Accuracy : 0.707         
##                  95% CI : (0.695, 0.718)
##     No Information Rate : 0.298         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.629         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.787    0.670    0.584    0.648    0.844
## Specificity             0.929    0.911    0.929    0.948    0.918
## Pos Pred Value          0.825    0.622    0.673    0.741    0.616
## Neg Pred Value          0.911    0.926    0.899    0.921    0.974
## Prevalence              0.298    0.180    0.201    0.187    0.134
## Detection Rate          0.235    0.120    0.117    0.121    0.113
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.858    0.791    0.756    0.798    0.881
```

```r
cm_qda
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1583   66   12   10    3
##          B   89  924  114    5    7
##          C    0   50  969    4    3
##          D    6    2  122  820   14
##          E    1   29   54   28  970
## 
## Overall Statistics
##                                         
##                Accuracy : 0.895         
##                  95% CI : (0.887, 0.903)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.867         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.943    0.863    0.762    0.946    0.973
## Specificity             0.978    0.955    0.988    0.971    0.977
## Pos Pred Value          0.946    0.811    0.944    0.851    0.896
## Neg Pred Value          0.977    0.969    0.938    0.990    0.994
## Prevalence              0.285    0.182    0.216    0.147    0.169
## Detection Rate          0.269    0.157    0.165    0.139    0.165
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.961    0.909    0.875    0.959    0.975
```

```r
cm_fda
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1539   67   34   34    0
##          B   90  884  130   29    6
##          C    0   99  892   34    1
##          D   14   21  106  813   10
##          E    2   70   78   45  887
## 
## Overall Statistics
##                                         
##                Accuracy : 0.852         
##                  95% CI : (0.843, 0.861)
##     No Information Rate : 0.28          
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.813         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.936    0.775    0.719    0.851    0.981
## Specificity             0.968    0.946    0.971    0.969    0.961
## Pos Pred Value          0.919    0.776    0.869    0.843    0.820
## Neg Pred Value          0.975    0.946    0.928    0.971    0.996
## Prevalence              0.280    0.194    0.211    0.162    0.154
## Detection Rate          0.262    0.150    0.152    0.138    0.151
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.952    0.861    0.845    0.910    0.971
```

```r
cm_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1644   16   13    0    1
##          B   43 1067   28    1    0
##          C    0   42  970   13    1
##          D    0    5   37  910   12
##          E    0   12    9   14 1047
## 
## Overall Statistics
##                                         
##                Accuracy : 0.958         
##                  95% CI : (0.953, 0.963)
##     No Information Rate : 0.287         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.947         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.975    0.934    0.918    0.970    0.987
## Specificity             0.993    0.985    0.988    0.989    0.993
## Pos Pred Value          0.982    0.937    0.945    0.944    0.968
## Neg Pred Value          0.990    0.984    0.982    0.994    0.997
## Prevalence              0.287    0.194    0.180    0.159    0.180
## Detection Rate          0.279    0.181    0.165    0.155    0.178
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.984    0.960    0.953    0.980    0.990
```


# Plot of accuracies

In the following plot, accuracy of each method will be displayed together with its confidence interval.

Two methods stands out from the rest and are virtually indistinguishable one other:
Random Forest and C5 with an accuracy of 99,3% and a very narrow confidence interval (0.4% wide).

Other 2 well performing techniques are the KNN and GBM with an accuracy larger than 96% but a wider confidence interval with respect to RF and C5.

The discriminant analysis models performance are much worse with accuracies lower than 90%. The LDA is especially poor since it does not even reach the 70% accuracy.



```r
Models <- data.frame(name = character(), accuracy = numeric(), lower = numeric(), 
    upper = numeric(), system_time = numeric(), user_time = numeric(), elapsed_time = numeric(), 
    stringsAsFactors = FALSE)

Models <- data.frame(name = "Random forests", accuracy = cm_rf$overall[1], lower = cm_rf$overall[3], 
    upper = cm_rf$overall[4], elapsed_time = rf_time[3], user_time = rf_time[1] + 
        rf_time[4], system_time = rf_time[2] + rf_time[5])

Models <- rbind(Models, data.frame(name = "KNN", accuracy = cm_knn$overall[1], 
    lower = cm_knn$overall[3], upper = cm_knn$overall[4], elapsed_time = knn_time[3], 
    user_time = knn_time[1] + knn_time[4], system_time = knn_time[2] + knn_time[5]))

Models <- rbind(Models, data.frame(name = "C5", accuracy = cm_c5$overall[1], 
    lower = cm_c5$overall[3], upper = cm_c5$overall[4], elapsed_time = c5_time[3], 
    user_time = c5_time[1] + c5_time[4], system_time = c5_time[2] + c5_time[5]))

Models <- rbind(Models, data.frame(name = "LDA", accuracy = cm_lda$overall[1], 
    lower = cm_lda$overall[3], upper = cm_lda$overall[4], elapsed_time = LDA_time[3], 
    user_time = LDA_time[1] + LDA_time[4], system_time = LDA_time[2] + LDA_time[5]))

Models <- rbind(Models, data.frame(name = "QDA", accuracy = cm_qda$overall[1], 
    lower = cm_qda$overall[3], upper = cm_qda$overall[4], elapsed_time = QDA_time[3], 
    user_time = QDA_time[1] + QDA_time[4], system_time = QDA_time[2] + QDA_time[5]))

Models <- rbind(Models, data.frame(name = "FDA", accuracy = cm_fda$overall[1], 
    lower = cm_fda$overall[3], upper = cm_fda$overall[4], elapsed_time = FDA_time[3], 
    user_time = FDA_time[1] + FDA_time[4], system_time = FDA_time[2] + FDA_time[5]))

Models <- rbind(Models, data.frame(name = "GBM", accuracy = cm_GBM$overall[1], 
    lower = cm_GBM$overall[3], upper = cm_GBM$overall[4], elapsed_time = GBM_time[3], 
    user_time = GBM_time[1] + GBM_time[4], system_time = GBM_time[2] + GBM_time[5]))
```



```r
Models <- Models[order(Models[, 2]), ]
ggplot(Models, aes(x = name, y = accuracy)) + geom_point(size = 4) + geom_errorbar(aes(ymin = lower, 
    ymax = upper), width = 0.2) + scale_x_discrete(limits = Models$name) + coord_flip()
```

![plot of chunk accuracy_plot](figure/accuracy_plot.png) 


# Plot of elapsed training time
During the model building phase, the function pre_proc() was used to calculate the time needed to build each of the 7 models. 

In the graph below, the training elapsed time of each model will be plot versus the accuracy, so that in 1 single plot we have both the variables of interest for models comparison and evaluation.

As you can see, Random Forest is an extremely resource consuming algorithm. It took 6.060 sec to train the data set on my iMac 2.7 GHz Core i5 with 8 GB 1600 DDR3 memory.

The c5 algorithm, while having the same exact accuracy, takes less than 1/3 of time to train the same training data (1.600 sec).

C5 is there fore a better model than Random Forest at least on this data set since while reaching the same accuracy, it is also much faster.

Actually, depending on how often the user is called to update the model building phase on, for instance, an updated data set, one may also consider to use the KNN algorithm instead, because while not being as good as c5 and random forest in terms of accuracy on the cross validation data (96,4% vs 99,3%), it takes only 131 sec to train the data set (vs 6.060sec and 1.600 sec of RF and C5 algorithms)




```r

rf_time
```

```
##    user  system elapsed 
## 5910.66   85.53 6006.97
```

```r
c5_time
```

```
##    user  system elapsed 
## 1571.85   13.27 1588.69
```

```r
knn_time
```

```
##    user  system elapsed 
## 123.688   4.668 128.359
```

```r
GBM_time
```

```
##    user  system elapsed 
## 525.480   3.981 529.541
```

```r
QDA_time
```

```
##    user  system elapsed 
##  15.459   2.178  17.637
```

```r
FDA_time
```

```
##    user  system elapsed 
##  511.08   11.52  522.69
```

```r
LDA_time
```

```
##    user  system elapsed 
##  16.842   2.578  19.746
```




```r
ggplot(Models, aes(x = accuracy, y = elapsed_time)) + geom_point() + geom_text(aes(x = accuracy, 
    label = name), size = 4, vjust = -1.5) + xlim(0.6, 1.1)
```

![plot of chunk elapsed_time_plot](figure/elapsed_time_plot.png) 

```r

```




# PERFORMANCE ON THE TEST DATA
As last step in this project, I would like to compare performance of random forest, c5 and knn models on the 20 row data set provided in the project both in terms of accuracy and in terms of testing elapsed time. Please notice that in this case, the elapsed time being considered is NOT the time to build the model (as above) but the time needed to make 20 predictions using a model that has already been built.

This test data will be used only once and, regardless the output, the models built will NOT be changed since this would invalidate the "independence" of the test set from the training procedure.

Before running the code, and before seeing the output I anticipate my expectations: if my analysis above is correct, I would expect the RF and c5 models giving the exact same output. I would expect also KNN giving the exact same result (knn accuracy is not as good as c5 but it is still very high) even though I would not be surprised for a few discrepancies.

Finally I would also expect knn testing time to be much lower than Random Forest and c5.



```r

testingData <- read.csv("pml-testing.csv", header = TRUE, na.strings = c("NA", 
    ""))

ptm <- proc.time()
test_rf <- predict(rf, newdata = testingData)
rf_test_time <- proc.time() - ptm

ptm <- proc.time()
test_c5 <- predict(c5, newdata = testingData)
c5_test_time <- proc.time() - ptm

ptm <- proc.time()
test_knn <- predict(knn, newdata = testingData)
knn_test_time <- proc.time() - ptm


test_rf
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
test_c5
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
test_knn
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



Let's count the output difference between RF/C5 and C5/knn.  
Number of different predicted cases between RF and c5: 

```r

sum(!(test_rf == test_c5))
```

```
## [1] 0
```


Number of different predicted cases between c5 and KNN: 

```r
sum(!(test_knn == test_c5))
```

```
## [1] 0
```


Differences in testing time: 

```r

rf_test_time
```

```
##    user  system elapsed 
##   0.521   0.516   1.431
```

```r
c5_test_time
```

```
##    user  system elapsed 
##   0.080   0.033   0.146
```

```r
knn_test_time
```

```
##    user  system elapsed 
##   0.127   0.015   0.154
```



C5 was 9.8014 faster than RF in making predictions.
Knn was 0.9481 faster than c5 in making predictions.


# COCLUSIONS

We have seen that while Random forest and c5 are the best model on this data set and virtually indistinguishable each other, c5 is much better than random forest since it reaches the same results in a much shorter time. c5 is therefore chosen as predferred model on this data set.

knn may become a excellent alternative if training elapsed time is a concern:  giving up for a few points in accuracy, substantial gaining in training and testing processing times can be achieved.

