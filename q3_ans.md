# Multi Class Logistic Regresssion

## Simple Regression

-- Accuracy of 0.9165275459098498 

## Autograd Regression

-- Accuracy of 0.885920979410128

## 4-Fold Validation and Confusion Matrix

![alt text](./logisticRegression/plots/q3_ConfusionMatrix.png?raw=true)

-- Two digits getting most confused are 8 and 1 with maximum off diagonal value of 20. Truth value is 8 and the wrong prediction is 1

-- The most easiest digits to predict are 0, 6 and 7 with 176 as the diagonal value.

## PCA with reduction to 2 dimensions

![alt text](./logisticRegression/plots/q3_PCA_plot.png?raw=true)

-- We can infer that the digits whose colors are overlapping are difficult to distinguish.
-- 0 and 6 are almost completely separated from other digits (easy to predict)
-- overlap can be seen between 1 and 8 as well as between 1 and 9.