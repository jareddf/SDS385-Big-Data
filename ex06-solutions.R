# ex06 solutions
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex06-functions.R')
library(xtable)
library(glmnet)

###
### 2b ---------------------
###


### Load data
y = read.csv('diabetesY.csv',header=F)[,1]
X = read.csv('diabetesX.csv')

# Standardize (center and scale)
y = scale(y)
X = scale(X[,1:10])



### Compare my algorithm to R's glmnet
# the effect of lambda differs. So let's look cv of lambda:

#seperate test/train
n = length(y)
train = sample(1:n,round(.8*n))
X.train = X[train,]
y.train = y[train,]
X.test = X[-train,]
y.test = y[-train]



# Choose grid of lambdas:
start.lambda = exp(seq(-4,4,.1))
mod.r = glmnet(x = X.train, y=y.train, family=c("gaussian"), alpha = 1, intercept = FALSE,lambda=start.lambda)

# lambda adjustment
lambda = mod.r$lambda * length(train)

# vector to store objective values for each lambda:
objective.jared <- objective.jared2 <- objective.r <- lambda * 0 


# Loop over lambda values, fit model, and calculate value of objective function
for(i in 1:length(lambda)){
  
  # R's glmnet
  objective.r[i] = sum((y.test - X.test%*%mod.r$beta[,i])^2) + mod.r$lambda[i]*sum(abs(mod.r$beta[,i]))
  
  # my pgd
  mod.jared = lasso.proximalgradient(y=y.train, X=X.train, lambda=lambda[i], stepsize.function=function(...){.001}, max.iter=100000, tol=.01 ,C=1/100,alpha=.9)
  objective.jared[i] = sum((y.test - X.test%*%mod.jared$beta)^2) + lambda[i]*sum(abs(mod.jared$beta))
  
  # my apgd
  mod.jared = lasso.accelerated.proximalgradient(y=y.train, X=X.train, lambda=lambda[i], stepsize.function=function(...){.001}, max.iter=100000, tol=.01 ,C=1/100,alpha=.9)
  objective.jared2[i] = sum((y.test - X.test%*%mod.jared$beta)^2) + lambda[i]*sum(abs(mod.jared$beta))
}



### Plot

#pdf('ex06_2b.pdf',height=4,width=6)
plot(log(lambda),objective.r,type='l'
     ,ylim = range(c(objective.r,objective.jared,objective.jared2))
     ,ylab = 'Objective Funtion Value'
     ,main = "Cross-Validation of Lasso's L1 Penalty"
     ,lwd=2
     )
lines(log(lambda),objective.jared,col=2,lty=2,lwd=2)
lines(log(lambda),objective.jared2,col=3,lty=3,lwd=3)
legend('topleft',legend=c("R's glmnet","My Proximal GD","My Accelerated PGD"),col=1:3,lty=1:3,lwd=3)
#dev.off()
#



### print "best" beta's
best.lambda = which.min(objective.r)

# my pgd
mod.jared = lasso.proximalgradient(y=y.train, X=X.train, lambda=lambda[best.lambda], stepsize.function=step.RobbinsMonro, max.iter=100000, tol=1 ,C=1/100,alpha=.9)

# my apgd
mod.jared2 = lasso.accelerated.proximalgradient(y=y.train, X=X.train, lambda=lambda[best.lambda], stepsize.function=step.RobbinsMonro, max.iter=100000, tol=1 ,C=1/100,alpha=.9)

print(cbind(mod.r$beta[,best.lambda],mod.jared$beta,mod.jared2$beta))


###
### 2c ---------------------------------------------------------------------------------
###
# Compare convergence:

lambda = .0001
C = 1/10
alpha = .99

# my pgd
mod.pgd = lasso.proximalgradient(y=y, X=X, lambda=lambda, stepsize.function=function(...){.001}, max.iter=100000, tol=1 ,C=C,alpha=alpha)

# my apgd
mod.apgd = lasso.accelerated.proximalgradient(y=y, X=X, lambda=lambda, stepsize.function=function(...){.001}, max.iter=100000, tol=.001)


#pdf('ex06_2c.pdf',height=4,width=6)
plot(mod.apgd$objective,type='l'
     ,col = 3
     ,ylab = 'Objective Funtion Value'
     ,main = "Convergence"
     ,lwd=2
     #,ylim=c(210,220)
)
lines(mod.pgd$objective,col=2,lwd=2)
legend('topleft',legend=c("My Proximal GD","My Accelerated PGD"),col=2:3,lty=c(1,3),lwd=3)
#dev.off()


# 
# # Comparison table
# mod.jared = lasso.proximalgradient(y=y, X=X, lambda=lambda[1], stepsize.function=step.RobbinsMonro, max.iter=100000, tol=10 ,C=1/100,alpha=.9)
# xtable(cbind(c(mod.r$a0,mod.r$beta[[1]]), mod.jared$beta))
#        





### Error testing:
# 
# 
# lasso.proximalgradient(
#   y=y
#   X=X
#   lambda=1
#   stepsize.function=function(){1}
#   max.iter=1000
#   min.iter=5
#     tol=.01

    