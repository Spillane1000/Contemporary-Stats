#install.packages("MASS")
library(MASS)

set.seed(123)
 ############################## Part (a) ####################
# Covariance Matrices
sigma1 = rbind(c(1,0.8),c(0.8,1))
sigma2 = rbind(c(1,-0.7),c(-0.7,1))

# Random samples from multivariate norm
N1 = mvrnorm(50, mu=c(0,0), Sigma=sigma1)
N2 = mvrnorm(50, mu=c(0,0), Sigma=sigma2)

# Creates classes
class1 = c(rep(1,50))
class2 = c(rep(2,50))

N1 = cbind(N1,class1) # joins classes to samples
N2 = cbind(N2,class2)

total_data = data.frame(rbind(N1,N2)) # full data set


colnames(total_data) = c("X1","X2","class")
total_data$class = as.factor(total_data$class) # transforms the class to a factor


####################### Part (b) ##########################

# learning LDA and QDA models based on data created 
model_lda = lda(class~.,total_data)
model_qda = qda(class~.,total_data)

# outputs the coefficients of discriminant function for the LDA and QDA
model_lda$scaling
model_qda$scaling

#### OUTPUT ##
#LD1
#X1  0.5983487
#X2 -0.8598253

#QDA 

#1         2
#X1 1.071166 -1.469346
#X2 0.000000  1.857198

#, , 2

#1        2
#X1 -1.022283 1.091010
#X2  0.000000 1.487222



###################### part (c) ############################
# create a testing set (the same as in part (a)
N1_test = mvrnorm(50, mu=c(0,0), Sigma=sigma1)
N2_test = mvrnorm(50, mu=c(0,0), Sigma=sigma2)

class1_test = c(rep(1,50))
class2_test = c(rep(2,50))

N1_test = cbind(N1_test,class1_test) # joins classes to samples
N2_test = cbind(N2_test,class2_test)

test_data = data.frame(rbind(N1_test,N2_test))
colnames(test_data) = c("X1","X2","class")
test_data$class = as.factor(test_data$class)

# predict classes of test data using LDA and QDA
lda_predictions = predict(model_lda,newdata = test_data[c("X1","X2")])$class
qda_predictions =  predict(model_qda,newdata = test_data[c("X1","X2")])$class 

#append predictions to test data
test_data = cbind(test_data, lda_predictions,qda_predictions)

# indicators for instances which have been miss-classified by LDA and QDA
test_data$lda_misclassified = abs(as.numeric(test_data$class) - as.numeric(test_data$lda_predictions)) 
test_data$qda_misclassified = abs(as.numeric(test_data$class) - as.numeric(test_data$qda_predictions))


# prints out miss-classification rates for both LDA and QDA models 
#Note: QDA consistently outperforms LDA for this taks
print(paste("lda miss classification rate: ", mean(test_data$lda_misclassified)*100,"%"))
print(paste("qda miss classification rate: ", mean(test_data$qda_misclassified)*100,"%"))

# OUTPUT
# lda miss classification rate:  46 %
# qda miss classification rate:  19 %

               