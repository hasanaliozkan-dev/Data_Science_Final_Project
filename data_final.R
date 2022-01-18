library(tidyverse)
library(cowplot)
library(reshape2)
library(caret)
library(Amelia)
library(corrplot)
library(mice)
library(VIM)
library(ROSE)
library(smotefamily)
library(party)
library(rpart.plot)
library(factoextra)
library(NbClust)

setwd("/Users/hasanaliozkan/Desktop/Desktop/")
#reading data
data <- read.csv("bank-additional-full.csv",sep=";")
data[data=="unknown"]<-NA
#viewving data
View(data)
str(data)
#setting categorical as a factor
data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$month <- as.factor(data$month)
data$day_of_week <- as.factor(data$day_of_week)
data$poutcome <- as.factor(data$poutcome)
data$y <- ifelse(data$y=="yes",1,0)
data$term_deposit <- as.factor(data$y)
data <- data[-21]

str(data)

label <- select(data,term_deposit)

#extracting sub_data
sub_data <- subset(data, select = - c(job,marital,education,default,housing,loan,contact,month,day_of_week,poutcome,term_deposit))
#Looking for missignes
missmap(data)

correlations <- cor(sub_data)

COL2(diverging = c("RdBu", "BrBG", "PiYG", "PRGn", "PuOr", "RdYlBu"), n = 200)

corrplot(correlations, is.corr = FALSE, method = 'color', col.lim = c(-1, 1),
         col = COL2('RdBu'), cl.pos = 'b', addCoef.col = 'black')

ggplot(data) +
  geom_point(mapping=aes(x = age, y = cons.price.idx, color= job))+
  labs(
    x = "Age (y)",
    y = "Talking Duration (s)",
    title = "Talking Durations of Customers of Different Ages")

ggplot(data) +
  geom_point(mapping=aes(x = age, y = duration, color= term_deposit))+
  labs(
    x = "Age (y)",
    y = "Talking Duration (s)",
    title = "Talking Durations of Customers of Different Ages")


set.seed(42)

data <- na.omit(data)

partial_data <- createDataPartition(y=data$term_deposit,p=0.8,list = FALSE)

train_data <- data[partial_data,]
test_data <- data[-partial_data,]

model <- glm(term_deposit~.,data = train_data,family = "binomial")

prob <- predict(model , newdata = test_data)
pred <- ifelse(prob>0.5 , 1,0)
mean(test_data$term_deposit == pred)

confusionMatrix(test_data$term_deposit,as.factor(pred))



real_value <- as.factor(pred)
predicted_value <- test_data$term_deposit

pred_vs_real <- as.data.frame(cbind(predicted_value,real_value))
real_value_vs_predicted_value <- as.data.frame(table(predicted_value,real_value))
# new plot will be added
plot(real_value_vs_predicted_value)

pca <- prcomp(sub_data,center = TRUE, scale = TRUE)
summary(pca)


missing_columns <- subset(data,select = c(default,education,loan,housing,job,marital))
summary(missing_columns)


missignes_plot <-aggr(missing_columns, col=c('lightblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(missing_columns), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern")) 
missignes_plot


summary(missing_columns)
impute <- mice(missing_columns,maxit = 0)

meth <- impute$method

meth[c("default")] <- "logreg"
meth[c("loan")] <- "logreg"
meth[c("housing")] <- "logreg"
meth[c("education")] <- "polyreg"
meth[c("job")] <- "polyreg"
meth[c("marital")] <- "polyreg"

#imputation 
set.seed(42)
imputed <-  mice(missing_columns,method = meth,m=5)

imputed <- complete(imputed)
imputed

summary(imputed)

#default variable is unbalanced.!!!!!!


#Looking for number of missing data.
sapply(imputed,function(x) sum(is.na(x)))

data$education <- imputed$education
data$default <- imputed$default
data$marital <- imputed$marital
data$loan <- imputed$loan
data$housing <- imputed$housing
data$job <- imputed$job


summary(data)
#OHE

cat_data <- subset(data,select=c(job,marital,education,default,housing,loan,contact,month,day_of_week,poutcome))
dummies_model <- dummyVars(~.,cat_data)
trainData_mat <- predict(dummies_model, newdata = cat_data)
summary(trainData_mat)

non_cat_data <- subset(data,select=-c(job,marital,education,default,housing,loan,contact,month,day_of_week,poutcome))
summary(non_cat_data)

result_data <- cbind(trainData_mat,non_cat_data)
summary(result_data)
ncol(result_data)

for_pca <- subset(result_data,select = -term_deposit)
summary(for_pca)
#there are a lot of column we definitely need pca


#first 29 PCA will taken
partial_data <- createDataPartition(y=result_data$term_deposit,p=0.8,list = FALSE)


train_data <- result_data[partial_data,]
test_data <- result_data[-partial_data,]
model <- glm(term_deposit~.,data = train_data,family = "binomial")


prob <- predict(model , newdata = test_data)
pred <- ifelse(prob>0.5 , 1,0)
pred
mean(test_data$term_deposit == pred)

confusionMatrix(test_data$term_deposit,as.factor(pred))
pca <- prcomp(train_data[1:57],center = TRUE, scale = TRUE)
summary(pca)
correlations <- cor(for_pca)
ncol(train_data)
COL2(diverging = c("RdBu", "BrBG", "PiYG", "PRGn", "PuOr", "RdYlBu"), n = 200)


corrplot::corrplot(correlations,method = "square",tl.cex = 0.5, tl.col = "black")
#to visualize whic components will take 
screeplot(pca, type = "lines", npcs = 50, main = "Screeplot of the first 15 PCs")
abline(h = 1, col="red", lty=5)
abline(v=29,col="blue",lty=4)
legend("topright", legend=c("Eigenvalue = 1","The last PC that have greater than eigen value=1"),
       col=c("red","blue"), lty=5, cex=0.6)


cumpro <- cumsum(pca$sdev^2 / sum(pca$sdev^2))
plot(cumpro[0:50], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 29, col="red", lty=5)
abline(h = 0.834, col="red", lty=5)
legend("topleft", legend=c("Cut-off @ PC29"),
       col=c("red"), lty=5, cex=0.6)



correlations <- cor(pca$x[,c(1:29)])

corrplot::corrplot(correlations,method = "square", tl.col = "black")
set.seed(42)

data_pca <- data.frame(term_deposit=train_data[,"term_deposit"],pca$x[,0:29])
head(data_pca)
pca$x

set.seed(42)
model_pca <- glm(term_deposit ~ .,data= data_pca,family = binomial)

test_data_pca <- predict(pca,newdata = test_data)

prob <- predict(model_pca , newdata = data.frame(test_data_pca[,1:29]),type = "response")

pred <- factor(ifelse(prob>0.5,1,0))

confusionMatrix(test_data$term_deposit,as.factor(pred))

#dengesiz olduğunu gösteren plot gelecek

table(result_data$term_deposit)

prop.table(table(result_data$term_deposit))

#Random over sampling

n_legit <- 36548
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit

job_blue <- subset(result_data,select = `job.blue-collar`)
names(result_data)[names(result_data)=="job.blue-collar"] <- "job_blue"
names(result_data)[names(result_data)=="job.self-employed"] <- "job_self"
nrow(result_data)

oversampling_result <- ovun.sample(term_deposit ~ .,
                                   data = result_data,
                                   method = "over",
                                   N = new_n_total,
                                   seed = 42)


table(oversampling_result$data$term_deposit)
#dengeye gelmiş halini çizdir

#burda  glm i tekrar et
partial_data <- createDataPartition(y=oversampling_result$data$term_deposit,p=0.8,list = FALSE)

train_data <- oversampling_result$data[partial_data,]
test_data <- oversampling_result$data[-partial_data,]
model <- glm(term_deposit~.,data = train_data,family = "binomial")

prob <- predict(model , newdata = test_data)
pred <- ifelse(prob>0.5 , 1,0)
mean(test_data$term_deposit == pred)

confusionMatrix(test_data$term_deposit,as.factor(pred))

#accuracy düştü fakat negatif prediction value arttı
#SMOTE

smote_output <- SMOTE(X = result_data[,-58],
                       target = result_data$term_deposit,
                       K = 5,
                       dup_size = 6)
oversampled_data = smote_output$data


table(oversampled_data$class)

oversampled_data$class <- as.factor(oversampled_data$class)

#dengeye gelmiş halini çizdir

#burda  glm i tekrar et
partial_data <- createDataPartition(y=oversampled_data$class,p=0.8,list = FALSE)

train_data <- oversampled_data[partial_data,]
test_data <- oversampled_data[-partial_data,]
model <- glm(class~.,data = train_data,family = "binomial")
class(oversampled_data$class)
prob <- predict(model , newdata = test_data)
pred <- ifelse(prob>0.5 , 1,0)
mean(test_data$class == pred)
confusionMatrix(test_data$class,as.factor(pred))

#Decision Tree
train_control_dTree <- trainControl(method = "repeatedcv",number = 10,repeats = 3)
set.seed(42)
decision_tree <- train(class~.,data = train_data,method="rpart", parms = list(split = "information"),
                       trControl=train_control_dTree,tuneLength = 10)
train_control_dTree
decision_tree


prp(decision_tree$finalModel, box.palette = "Reds", tweak = 1)

decision_tree$modelInfo
test_data[1,]


pred <- predict(decision_tree,newdata = test_data)
confusionMatrix(as.factor(pred),test_data$class)

#knn classification 
knn <- knn3(class~.,train_data,k=3)

#head(knn$learn$y)
#head(predict(knn, test_data, type = "class"), n = 10)
#head(predict(knn, test_data, type = "prob"), n = 10)



pred <- predict(knn,newdata = test_data,type = "class")
confusionMatrix(as.factor(pred),test_data$class)

#k means clustering 


model_knn <- kmeans(oversampled_data[,-58],centers = 2,nstart = 20)


fviz_cluster(model, oversampled_data[,-58],  palette = c("red", "green", "blue"),
             geom = "point",
             ellipse.type = "convex")
table(train_data$class,model$cluster)


            
fviz_nbclust(oversampled_data[0:6000,-58], kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2,col="red") + 
  labs(subtitle = "Elbow method") 

fviz_nbclust(oversampled_data[0:6000,-58], kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")

fviz_nbclust(oversampled_data[0:5000,-58], kmeans,
             nstart = 25,
             method = "gap_stat",
             nboot = 10 
) +
  labs(subtitle = "Gap statistic method")


#BSS-TSS

BSS <- model_knn$betweenss
TSS <- model_knn$totss
BSS/TSS * 100

model_knn <- kmeans(oversampled_data[,-58],centers = 3,nstart = 20)

BSS <- model_knn$betweenss
TSS <- model_knn$totss
BSS/TSS * 100





#hc clustering 
distance <- dist(oversampled_data[1:10000,-58],method = "euclidian")
model_hc <- hclust(distance,method = "average")



plot(model_hc)
# linear regression to do that we have to use ohe to term_deposit variable

#OHE
cat_data <- subset(oversampled_data,select=class)
dummies_model <- dummyVars(~.,cat_data)
trainData_mat <- predict(dummies_model, newdata = cat_data)
summary(trainData_mat)


data_for_regression <- cbind(oversampled_data,trainData_mat)
data_for_regression <- data_for_regression[,-58]

partial_data <- createDataPartition(y=data_for_regression$cons.conf.idx ,p=0.8,list = FALSE)

train_data <- oversampled_data[partial_data,]
test_data <- oversampled_data[-partial_data,]


lin_reg <- lm(cons.conf.idx~.,data =train_data)
print(lin_reg)
result <- predict(lin_reg,test_data)


summary(lin_reg)

#in summary there are 3 star variables that means they are 
                                              #important to predict our dependent variable.


data_for_regression_with_meaningful_columns <- subset(data_for_regression,
                                              select=c(job.technician,housing.no,contact.cellular,
                                                       month.apr,month.aug,month.dec,month.jul,month.jun 
                                                       ,month.mar,month.may,month.nov,month.oct,month.sep,day_of_week.fri,
                                                       day_of_week.thu ,age,pdays,emp.var.rate,
                                                       cons.price.idx,euribor3m ,cons.conf.idx))




partial_data <- createDataPartition(y=data_for_regression_with_meaningful_columns$cons.conf.idx ,p=0.8,list = FALSE)

train_data <- oversampled_data[partial_data,]
test_data <- oversampled_data[-partial_data,]
lin_reg <- lm(cons.conf.idx~.,data =train_data)
summary(lin_reg)


rect.hclust(hier, k=2, border="red")



