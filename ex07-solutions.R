#ex07-solutions.R

setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex06-ProximalGD/ex06-functions.R')
source('ex07-ADMM/ex07-functions.R')
library(xtable)
library(glmnet)



### Load data
y = read.csv('diabetesY.csv',header=F)[,1]
X = read.csv('diabetesX.csv')

# Standardize (center and scale)
y = scale(y)
X = scale(X[,1:10])


mod = admm_lasso(Y=y, X=X, lambda = 1, rho=.01, tol = .1, max.iter = 99999)
print(mod$beta)
plot(log(1:length(mod$objective)),mod$objective,type='l')

glmnet()