library(ggplot2)

# Check the data type of all the variables
sapply(Absenteeism_at_work,class)

# Change variable seasons into character
Absenteeism_at_work$Seasons <- as.character(Absenteeism_at_work$Seasons)
Absenteeism_at_work$Education <- as.character(Absenteeism_at_work$Education)
Absenteeism_at_work$Social.drinker <- as.character(Absenteeism_at_work$Social.drinker)
Absenteeism_at_work$Social.smoker <- as.character(Absenteeism_at_work$Social.smoker)
Absenteeism_at_work$Disciplinary.failure <- as.character(Absenteeism_at_work$Disciplinary.failure)

# Check the response varaible 
length(which(Absenteeism_at_work$Absenteeism.time.in.hours == 0))
Absenteeism_at_work$Reason.for.absence <- as.factor(Absenteeism_at_work$Reason.for.absence)
summary(Absenteeism_at_work$Absenteeism.time.in.hours)

# check the distribution of the response variable
g <- ggplot(Absenteeism_at_work,aes(x = Absenteeism_at_work$Absenteeism.time.in.hours))
g <- g + geom_histogram(breaks = seq(0,120,5))
print(g)

# Highly posiively skewed 
# Create a binary response variable
Absenteeism_at_work$Response_Binary <- ifelse(Absenteeism_at_work$Absenteeism.time.in.hours <=3,1,0)
summary(Absenteeism_at_work$Response_Binary)
length(which(Absenteeism_at_work$Response_Binary == 1))

length(which(Absenteeism_at_work$Absenteeism.time.in.hours == 2))
length(which(Absenteeism_at_work$Absenteeism.time.in.hours == 8))
length(which(Absenteeism_at_work$Absenteeism.time.in.hours > 8))
length(which(Absenteeism_at_work$Absenteeism.time.in.hours <= 2))
length(which(Absenteeism_at_work$Absenteeism.time.in.hours < 8 & Absenteeism_at_work$Absenteeism.time.in.hours >2))
length(which(Absenteeism_at_work$Absenteeism.time.in.hours >= 8))

# Create a three-class response variable
Absenteeism_at_work$Response_Class[Absenteeism_at_work$Absenteeism.time.in.hours <= 2] <- "2"
Absenteeism_at_work$Response_Class[Absenteeism_at_work$Absenteeism.time.in.hours < 8 & Absenteeism_at_work$Absenteeism.time.in.hours > 2] <- "1"
Absenteeism_at_work$Response_Class[Absenteeism_at_work$Absenteeism.time.in.hours >= 8] <- "0"
summary(Absenteeism_at_work$Response_Class)

# create dummy variables 
library(dummies)
reason <- Absenteeism_at_work$Reason.for.absence
reason <- cbind(reason,dummy(Absenteeism_at_work$Reason.for.absence, sep = "-"))
reason <- data.frame(reason)
reason

# Binary Class
reason.1 <- reason
reason.1$Class <- Absenteeism_at_work$Response_Binary
reason.1$reason <- NULL
reason.1$Class <- NULL

#Three level Class
reason.2 <- reason
reason.2$Class <- Absenteeism_at_work$Response_Class
reason.2$reason <- NULL

# run a clustering on the dummy variables for reason absent
library(klaR)
set.seed(05302018)
cluster.results <-kmodes(reason.1, 5, iter.max = 100, weighted = FALSE ) #don't use the record ID as a clustering variable!
sapply(cluster.results,class)

cluster.results$cluster

cluster.results

cluster.output <- cbind(data.to.cl,cluster.results$cluster)

cluster.results.cluster <- data.frame(cluster.results$cluster)
cluster.results.modes <- data.frame(cluster.results$modes)

write.csv(cluster.results.cluster, file = "cluster.results.cluster.csv")
write.csv(cluster.results.modes, file = "cluster.results.modes.csv")

pairs(reason.1[2:6], col=reason.1$Class, cex=0.5)

absent$BinaryClass <- Absenteeism_at_work$Response_Binary
#absent$ThreeClass <- Absenteeism_at_work$Response_Class
absent$Obs <- NULL
absent$ID <- NULL
absent$Reason.for.absence <- NULL
absent$Class <- Absenteeism_at_work$Response_Binary
#absent$Cluster <- NULL
#absent$ThreeClass <- NULL
absent$Absenteeism.time.in.hours <- NULL


# Feature 1 
absent.reasons <- absent$Cluster
absent.reasons <- cbind(absent.reasons,dummy(absent$Cluster, sep = "-"))
absent.reasons <- data.frame(absent.reasons)
absent.reasons$absent.reasons <- NULL

# Feature 2 
absent.month <- absent$Month.of.absence
absent.month <- cbind(absent.month,dummy(absent$Month.of.absence, sep = "-"))
absent.month <- data.frame(absent.month)
absent.month$absent.month <- NULL
absent.month$absent.month.0 <- NULL

# Feature 3
absent.weekday <- absent$Day.of.the.week
absent.weekday <- cbind(absent.weekday,dummy(absent$Day.of.the.week, sep = "-"))
absent.weekday <- data.frame(absent.weekday)
absent.weekday$absent.weekday <- NULL

# Feature 4
absent.season <- absent$Seasons
absent.season <- cbind(absent.season,dummy(absent$Seasons, sep = "-"))
absent.season <- data.frame(absent.season)
absent.season$absent.season <- NULL

# Feature 11
absent.disp <- absent$Disciplinary.failure
absent.disp <- cbind(absent.disp,dummy(absent$Disciplinary.failure, sep = "-"))
absent.disp <- data.frame(absent.disp)
absent.disp$absent.disp <- NULL


# Feature 12
absent.education <- absent$Education
absent.education <- cbind(absent.education,dummy(absent$Education, sep = "-"))
absent.education <- data.frame(absent.education)
absent.education$absent.education <- NULL

# Feature 13
absent.child <- absent$Son
absent.child <- cbind(absent.child,dummy(absent$Son, sep = "-"))
absent.child <- data.frame(absent.child)
absent.child$absent.child <- NULL

# Feature 14
absent.drinker <- absent$Social.drinker
absent.drinker <- cbind(absent.drinker,dummy(absent$Social.drinker, sep = "-"))
absent.drinker <- data.frame(absent.drinker)
absent.drinker$absent.drinker <- NULL

# Feature 15
absent.smoker <- absent$Social.smoker
absent.smoker <- cbind(absent.smoker,dummy(absent$Social.smoker, sep = "-"))
absent.smoker <- data.frame(absent.smoker)
absent.smoker$absent.smoker <- NULL

# Feature 16 
absent.pet <- absent$Pet
absent.pet <- cbind(absent.pet,dummy(absent$Pet, sep = "-"))
absent.pet <- data.frame(absent.pet)
absent.pet$absent.pet <- NULL

# Class feature 
absent.class <- absent$Class
absent.class <- cbind(absent.class,dummy(absent$Class, sep = "-"))
absent.class <- data.frame(absent.class)
absent.class
absent.class$absent.class <- NULL


# Combine all binary features into 1 dataframe

binary.absent <- cbind(absent.reasons,absent.month,absent.weekday,absent.season,absent.disp,
                        absent.education,absent.child,absent.drinker,absent.smoker,absent.pet,absent.class)

# Biclustering

library("biclust")
library(psych)
help("biclust")

binary.matrix <- data.frame(absent)
str(binary.matrix)
binary.martix <- data.frame(lapply(binary.matrix, as.numeric))
binary.matrix <- as.matrix(absent)
loma <- binarize(binary.matrix)

set.seed(7)
binary.absent.martix <- as.matrix(binary.absent) # works
str(binary.absent)
loma <- binarize(binary.absent.martix) # works
res <- biclust(x=loma, method=BCXmotifs())
res
biclustmember(x = binary.absent.martix, bicResult = res, font = 112)

a <- writeclust(res,row=TRUE,noC=5)
col <- data.frame(a)
col
absent.bicluster <- absent 
absent.bicluster$BinaryClass <- NULL
absent.bicluster$Bicluster <- col$a
absent.bicluster$HoursMissed <- Absenteeism_at_work$Absenteeism.time.in.hours

cluster_hours <- subset(absent.bicluster, absent.bicluster$HoursMissed <= 5)
cluster_hours
summary(cluster_hours)
cluster_hoursles5 <- data.frame(describe(cluster_hours))
write.csv(cluster_hoursles5 , file = "cluster_hours5.csv") # these are 2nd 

cluster_more3 <- subset(absent.bicluster, absent.bicluster$HoursMissed > 5)
summary(cluster_more3)
cluster_hoursmore5 <-data.frame(describe(cluster_more3))
write.csv(cluster_hoursmore5, file = "cluster_more5.csv") # these are 2nd 


t.test(absent.bicluster$HoursMissed[absent.bicluster$Bicluster=="1"],absent.bicluster$HoursMissed[absent.bicluster$Bicluster=="2"])
t.test <- t.test(cluster_hours$HoursMissed[cluster_hours$Bicluster],cluster_more3$HoursMissed[cluster_more3$Bicluster])
t.test
ts = replicate(1000,t.test(rnorm(10),rnorm(10))$statistic)
pts = seq(-4.5,4.5,length=100)
plot(pts,dt(pts,df=18),col='red',type='l')
qqplot(ts,rt(1000,df=400))
abline(0,1)

t.test <- t.test(cluster_hours,cluster_more3)
t.test

cluster_1 <- subset(absent.bicluster, absent.bicluster$Bicluster == 1)
summary(cluster_1)
cluster_1des <- data.frame(describe(cluster_1)) 
write.csv(cluster_1des,  file = "cluster_1.csv")

length(which(cluster_1$Social.smoker == 1))
length(which(cluster_1$Social.smoker == 0))

length(which(cluster_1$Social.drinker == 1))
length(which(cluster_1$Social.drinker == 0))


cluster_2 <- subset(absent.bicluster, absent.bicluster$Bicluster == 2)
summary(cluster_2)
cluster_2des <- data.frame(describe(cluster_2))
lines(density(ts))
write.csv(cluster_2des,  file = "cluster_2.csv")



length(which(cluster_2$Social.smoker == 1))
length(which(cluster_2$Social.smoker == 0))

length(which(cluster_2$Social.drinker == 1))
length(which(cluster_2$Social.drinker == 0))


cluster_3 <- subset(absent.bicluster, absent.bicluster$Bicluster == 3)
summary(cluster_3)
cluster_3des <- data.frame(describe(cluster_3))
write.csv(cluster_3des,  file = "cluster_3.csv")


length(which(cluster_3$Social.smoker == 1))
length(which(cluster_3$Social.smoker == 0))

length(which(cluster_3$Social.drinker == 1))
length(which(cluster_3$Social.drinker == 0))


cluster_4 <- subset(absent.bicluster, absent.bicluster$Bicluster == 4)
summary(cluster_4)
cluster_4des <- describe(cluster_4)
write.csv(cluster_4des,  file = "cluster_4.csv")


length(which(cluster_4$Social.smoker == 1))
length(which(cluster_4$Social.smoker == 0))

length(which(cluster_4$Social.drinker == 1))
length(which(cluster_4$Social.drinker == 0))


cluster_5 <- subset(absent.bicluster, absent.bicluster$Bicluster == 5)
summary(cluster_5)
cluster_5des <-data.frame(describe(cluster_5))
write.csv(cluster_5des,  file = "cluster_5.csv")


length(which(cluster_5$Social.smoker == 1))
length(which(cluster_5$Social.smoker == 0))

length(which(cluster_5$Social.drinker == 1))
length(which(cluster_5$Social.drinker == 0))


# Log regression
options("scipen"= 100, "digits" = 4)
absent$Obs <- NULL
absent$ID <- NULL
absent$Reason.for.absence <- NULL
absent$Absenteeism.time.in.hours <- NULL
absent$Class <- Absenteeism_at_work$Response_Binary
str(absent)
absent$Class <- factor(absent$Class, levels = c("1","0"))

set.seed(5272018)

ind <- sample(2, nrow(absent), replace=TRUE, prob=c(0.8, 0.2))
train <- absent[ind==1,]
test <- absent[ind==2,]

plot(absent$Distance.from.Residence.to.Work,absent$Transportation.expense)
plot(absent$Service.time,absent$Age)

absent$Class <- factor(absent$Clas)

#binomial

fitAll <- glm(Class ~., data = train, family = "binomial")
summary(fitAll)
fit.1 <- glm(Class ~ 1, data = train, family = "binomial")
summary(fit.1)
fitted.results <- predict(fit.1,newdata=test,type='response')

t <- table(fitted.results,test$Class)
confusionMatrix(t)
mean(pred != test$Class) #.32


fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$Class)
print(paste('Accuracy',1-misClasificError))

forward <- step(fit.1, direction = 'forward',scope=list(upper=fitAll,lower=fit.1))

final.log <- glm(Class ~ Cluster + Disciplinary.failure + Transportation.expense + 
                   Social.drinker + Pet + Day.of.the.week + Son + Seasons, family = 'binomial',
                 data = train)

summary(final.log)
fitted.results <- predict(final.log,newdata=test,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
t <- table(fitted.results,test$Class)
confusionMatrix(t)
misClasificError <- mean(fitted.results != test$Class)
print(paste('Accuracy',1-misClasificError))


stepwise <- step(fit.1, direction = 'both',scope=list(upper=fitAll,lower=fit.1))

names(sig.features)

library(pscl)
pR2(final.log)

sig.features <- absent[,c(1,11,5,14,16,3,13,4)]  # features of logistical regression using forward selection
sig.features


# SVM
help(svm)
library(e1071)
library('caret')
absent$Obs <- NULL
absent$ID <- NULL
absent$Reason.for.absence <- NULL
absent$Absenteeism.time.in.hours <- NULL
absent$Height <- NULL
absent$Weight <- NULL
absent$Class <- Absenteeism_at_work$Response_Binary
absent$Class <- as.factor(absent$Class)

absent$Class <- as.character(absent$Class)
absent$Class[absent$Class == "0"] <- "-1"
absent$Class <- as.factor(absent$Class)

str(absent)

set.seed(5272018)
ind <- sample(2, nrow(absent), replace=TRUE, prob=c(0.8, 0.2))
train <- absent[ind==1,]
test <- absent[ind==2,]

sig.features$Class <- as.factor(absent$Class)
sig.features$Class <- as.character(sig.features$Class)
sig.features$Class[sig.features$Class == "0"] <- "-1"
sig.features$Class <- as.factor(sig.features$Class)
sig.features$Class <- factor(sig.features$Class, levels = c("1","-1"))
str(sig.features)


ind <- sample(2, nrow(sig.features), replace=TRUE, prob=c(0.8, 0.2))
train2 <- sig.features[ind==1,]
test2 <- sig.features[ind==2,]

# all features - polynomial
svm.features <- svm(Class ~., kernel = 'polynomial', data = train)
summary(svm.features)
pred <- predict(svm.features,train)
t <- table(pred,train$Class)
confusionMatrix(t)
mean(pred != train$Class) #.40


pred <- predict(svm.features,test)
t <- table(pred,test$Class)
confusionMatrix(t)
mean(pred != test$Class) #.34

# sig features - polynoimal 
svm.sig <- svm(Class ~., kernel = 'polynomial', data = train2)
summary(svm.sig)
pred <- predict(svm.sig,train2)
t <- table(pred,train2$Class)
confusionMatrix(t)
mean(pred != train2$Class) #.29

pred <- predict(svm.sig,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class) #.29


# all features - radial
svm.features <- svm(Class ~., kernel = 'radial', data = train)
summary(svm.features)
pred <- predict(svm.features,train)
t <- table(pred,train$Class)
confusionMatrix(t)
mean(pred != train$Class) #.40

pred <- predict(svm.features,test)
t <- table(pred,test$Class)
confusionMatrix(t)
mean(pred != test$Class) #.32


# sig features - radial
svm.features <- svm(Class ~., kernel = 'radial', data = train2)
summary(svm.features)

pred <- predict(svm.features,train2)
t <- table(pred,train2$Class)
confusionMatrix(t)
mean(pred != train2$Class) #28 

pred <- predict(svm.features,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class) #28 

# all features - sigmoid
svm.features <- svm(Class ~., kernel = 'sigmoid', data = train)
summary(svm.features)


pred <- predict(svm.features,train)
t <- table(pred,train$Class)
confusionMatrix(t)
mean(pred != train$Class) #.40

pred <- predict(svm.features,test)
t <- table(pred,test$Class)
confusionMatrix(t)
mean(pred != test$Class) #.40

# sig features - sigmoid
svm.sig <- svm(Class ~., kernel = 'sigmoid', data = train2)
summary(svm.sig)

pred <- predict(svm.sig,train2)
t <- table(pred,train2$Class)
confusionMatrix(t)
mean(pred != train2$Class) #28 

pred <- predict(svm.sig,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class) #.38


# all features - linear
svm.features <- svm(Class ~., kernel = 'linear', data = train)
summary(svm.features)
pred <- predict(svm.features,train)
t <- table(pred,train$Class)
confusionMatrix(t)
mean(pred != train$Class) #.40


pred <- predict(svm.features,test)
t <- table(pred,test$Class)
confusionMatrix(t)
mean(pred != test$Class) #.30

# sig features - linear
svm.sig <- svm(Class ~., kernel = 'linear', data = train2)
summary(svm.sig)

pred <- predict(svm.sig,train2)
t <- table(pred,train2$Class)
confusionMatrix(t)
mean(pred != train2$Class) #.28


pred <- predict(svm.sig,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class) #.28

# all features - sigmoid
svm.features <- svm(Class ~., kernel = 'sigmoid', data = train)
summary(svm.features)
pred <- predict(svm.features,test)
t <- table(pred,test$Class)
confusionMatrix(t)
mean(pred != test$Class) #.40

# sig features - sigmoid
svm.sig <- svm(Class ~., kernel = 'sigmoid', data = train2)
summary(svm.sig)
pred <- predict(svm.sig,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class) #.38

# tuned SVM using CV
mytunedsvm.poly <- tune.svm(Class ~ ., kernel = "polynomial", data = train2,cost = 10^(-10:-1), gamma = log10(seq(-5:5)))
summary(mytunedsvm.poly)
mytunedsvm.poly.perf <- data.frame(mytunedsvm.poly$performances)

mytunedsvm.poly.perf$logx <- log10(mytunedsvm.poly.perf$cost)
poly.graph <- ggplot(mytunedsvm.poly.perf,aes(x= mytunedsvm.poly.perf$logx,y= mytunedsvm.poly.perf$error))
poly.graph <- linear.graph + geom_line() 
poly.graph 


mytunedsvm.linear <- tune.svm(Class ~ ., kernel = "linear", data = train2, cost = 10^(-10:-1))
summary(mytunedsvm.linear)
mytunedsvm.linear.perf <- data.frame(mytunedsvm.linear$performances)
mytunedsvm.linear.perf

mytunedsvm.linear.perf$logx <- log10(mytunedsvm.linear.perf$cost)
linear.graph <- ggplot(mytunedsvm.linear.perf,aes(x= mytunedsvm.linear.perf$logx,y= mytunedsvm.linear.perf$error))
linear.graph <- radial.graph + geom_line() 
linear.graph 


mytunedsvm.radial <- tune.svm(Class ~ ., kernel = "radial", data = train2, cost = 10^(-10:1), gamma = log10(seq(-5:5)), coef0 = seq(-1,10,1) )
summary(mytunedsvm.radial)
mytunedsvm.radial.perf <- data.frame(mytunedsvm.radial$performances)
mytunedsvm.radial.perf
 

mytunedsvm.radial.perf$logx <- log10(mytunedsvm.radial.perf$cost)
radial.graph <- ggplot(mytunedsvm.radial.perf,aes(x= mytunedsvm.radial.perf$logx,y= mytunedsvm.radial.perf$error))
radial.graph <- radial.graph + geom_line() 
radial.graph 


# final svm model

svm.poly <- svm(Class ~ ., kernel = "polynomial", data = train2, cost = .1 ,gamma = 0.9542)
summary(svm.poly)

pred.train <- predict(svm.poly,train2)
t <- table(pred.train,train2$Class)
confusionMatrix(t)
mean(pred.train != train2$Class)


pred <- predict(svm.poly,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class)

svm.linear <- svm(Class ~ ., kernel = "linear", data = train2,  cost = .01)
summary(svm.linear)

pred.train <- predict(svm.linear,train2)
t <- table(pred.train,train2$Class)
confusionMatrix(t)
mean(pred.train != train2$Class)

pred <- predict(svm.linear,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class)



svm.radial <- svm(Class ~ ., kernel = "radial", data = train2,  cost = 1, gamma = 0.7782, coef0 = -1)
summary(svm.radial)
svm.radial.per <- data.frame(svm.radial)


radial.perf$logx <- log10(radial.perf$gamma)
radial.graph <- ggplot(radial.perf,aes(x=radial.perf$logx,y=radial.perf$error))
radial.graph <- radial.graph + geom_line() 
radial.graph 


pred.train <- predict(svm.radial,train2)
t <- table(pred.train,train2$Class)
confusionMatrix(t)
mean(pred.train != train2$Class)


pred <- predict(svm.radial,test2)
t <- table(pred,test2$Class)
confusionMatrix(t)
mean(pred != test2$Class)



svm.radial.final <- svm(Class ~ ., kernel = "radial", data = sig.features,  cost = 1, gamma = 0.7782, coef0 = -1)
summary(svm.radial.final)


pred <- predict(svm.radial.final,sig.features)
t <- table(pred,sig.features$Class)
confusionMatrix(t)
mean(pred != sig.features$Class)


# Probabilities for ROC
require(ROCR)

poly.prob.svm <- svm(Class~., data = train2,kernel = 'polynomial', gamma = 0.9542, cost = .1 ,probability = TRUE)
summary(poly.prob.svm)
poly.svm.pred <- predict(poly.prob.svm, type="prob", newdata=test2, probability = TRUE)

poly.svm.prob.rocr <- prediction(attr(poly.svm.pred, "probabilities")[,2], test2$Class)
poly.svm.perf <- performance(poly.svm.prob.rocr, "tpr","fpr")
plot(poly.svm.perf,main="ROC curve for Polynomial Kernel")
abline(a=0,b=1)

linear.prob.svm <- svm(Class~., data = train2,kernel = 'linear', cost= 1, gamma=0.003906, probability = TRUE)
summary(linear.prob.svm)
linear.svm.pred <- predict(linear.prob.svm, type="prob", newdata=test2, probability = TRUE)

linear.svm.prob.rocr <- prediction(attr(linear.svm.pred, "probabilities")[,2], test2$Class)
linear.svm.perf <- performance(linear.svm.prob.rocr, "tpr","fpr")
plot(linear.svm.perf,main="ROC curve for Linear Kernel")
abline(a=0,b=1)

radial.prob.svm <- svm(Class~., data = train2,kernel = 'linear', cost= 1, gamma=0.7782, coef0 = -1, probability = TRUE)
summary(radial.prob.svm)
radial.svm.pred <- predict(radial.prob.svm, type="prob", newdata=test2, probability = TRUE)

radial.svm.prob.rocr <- prediction(attr(radial.svm.pred, "probabilities")[,2], test2$Class)
radial.svm.perf <- performance(radial.svm.prob.rocr, "tpr","fpr")
plot(radial.svm.perf,main="ROC curve for Radial Kernel")
abline(a=0,b=1)
