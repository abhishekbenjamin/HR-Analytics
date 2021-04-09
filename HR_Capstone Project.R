hr <- read.csv("C:/Users/Abhishek Benjamin/Desktop/Download -2/Statistics/Capstone Project/HR_Employee_Attrition_Data.csv", stringsAsFactors = T)


library(caret)
library(randomForest)
library(Boruta)
library(DMwR)

str(hr)

## Handling Class Imbalance

table(hr$Attrition)

smoted_data <- SMOTE(Attrition ~., data = hr, perc.over=100)

table(smoted_data$Attrition)


## Feature selection

set.seed(123)

b1 <- Boruta(Attrition ~., data = smoted_data, doTrace = 2, maxRuns = 15)

print(b1)

## Data Partition

set.seed(123)

pd <- sample(2, nrow(smoted_data), replace = T, prob = c(0.6,0.4))

train <- smoted_data[pd==1,]
test <- smoted_data[pd==2,]

## Fitting Random Forest

set.seed(1234)

rf34 <- randomForest(Attrition ~., data = train)

## Prediction & Confusion Matrix

pred <- predict(rf34, test)

confusionMatrix(pred,test$Attrition)

mis_1 <- table(pred, test$Attrition)
mis_1
1- sum(diag(mis_1))/sum(mis_1)

getConfirmedFormula(b1)

rf31 <- randomForest( Attrition ~ Age + BusinessTravel + DailyRate + Department + DistanceFromHome + 
                        Education + EducationField + EmployeeNumber + EnvironmentSatisfaction + 
                        Gender + HourlyRate + JobInvolvement + JobLevel + JobRole + 
                        JobSatisfaction + MaritalStatus + MonthlyIncome + MonthlyRate + 
                        NumCompaniesWorked + OverTime + PercentSalaryHike + PerformanceRating + 
                        RelationshipSatisfaction + StockOptionLevel + TotalWorkingYears + 
                        TrainingTimesLastYear + WorkLifeBalance + YearsAtCompany + 
                        YearsInCurrentRole + YearsSinceLastPromotion + YearsWithCurrManager,data = train)


## New Prediction & Confusion Matrix

pred_2 <- predict(rf31, test)

confusionMatrix(pred_2, test$Attrition, mode = "prec_recall")

## Important Variables in Random Forest

varImpPlot(rf31, sort = TRUE, main = "Top 10 Important Variables", n.var = 10)

importance(rf31)

varUsed(rf31)



rm(list = ls())



