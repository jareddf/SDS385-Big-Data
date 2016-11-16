# ex09 execution script for solutions
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/ex09-MatrixFactorization/')
source('ex09-functions.R')

### 
### 1 ---------------------------------------------------------------
###

# Write your own code, following the exposition of Witten et. al., that is capable of solving this sparse rank-1 factorization problem for a given choice of λu and λv
'Sourced above'

# Simulate some data to make sure that your implementation is behaving sensibly. 

signal = rnorm(20)
x.sd = .5
X = cbind(signal + rnorm(20,0,x.sd), signal + rnorm(20,0,x.sd))


bob = PMD.singlefactor(X = X,cv = 5,cu = 5)
new.x =  as.numeric(bob$d) * bob$u %*% t(bob$v)

pdf('ex09_1.pdf')
plot(X,pch=19)
points(new.x[,1],new.x[,2],col=2,pch='x')
dev.off()

###
### 2 --------------------------------------------------------------
###

# Twitter/marketing data

dat = read.csv('../social_marketing.csv')
X = as.matrix(dat[,-1])

# Exploratory
pca = prcomp(X)
plot(pca)
plot(pca$sdev)

# k=1:5
out = PMD.multifactor(k=4,X=X,cv=1.5,cu=3,tol=.001,max.iter=10000)
library(xtable)
xtable(out$V,digits=4)


