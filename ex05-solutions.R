# Script for Exercises 05, James Scott's SDS 385 - Big Data
setwd('~/Dropbox/School/SDS385_Big_Data/')

### 
### 1A -----------------------------------------------------------------------------
###

# S_lambda function without the argmin
ell = function(y, theta, lambda){
  .5 * (y-theta)^2 + lambda * theta
}

# Give values of lambda and y
lambda = 1
y = 5 

#produce plot of function, showing 
curve(ell(y,x,lambda),-10,10,xlab=expression(theta),ylab=expression(S^1[lambda](5)))
yyy = abs(y) - lambda
abline(v=ifelse(y<0,-1,1)*ifelse(yyy>0,yyy,0),col=2)

# Hard thresholding function. 
H_lambda = function(y,lambda){
  ifelse(abs(y)>=lambda,y,0)
}

# Soft thresholing function
S_lambda = function(y,lambda){
  yyy = abs(y) - lambda
  out = ifelse(y<0,-1,1)*ifelse(yyy>0,yyy,0)
  return(out)
}

# plot different thresholdings
pdf('ex05_1a.pdf',width=6,height=4)
curve(H_lambda(x,1),-3,3,n=10000,xlab='y',ylab=expression(hat(theta)))
curve(S_lambda(x,1),-3,3,n=10000,xlab='y',ylab=expression(hat(theta)),col=2,lty=3,add=T,lwd=3)
legend('topleft',legend = c('Hard','Soft'), col=1:2, lty=c(1,3), lwd=c(1,3))
title(expression(paste('Thresholding, ',lambda,'=1')))
dev.off()


###
### 1B --------------------------------------------------------------------------------------
###

# Number of datapoints
n = 100
density = .2 #percent 

### Step 1
# Sparse theta vector
n.nonzero = round(n*density)
theta = numeric(n)
theta[sample(1:n,n.nonzero)] <- 1:n.nonzero

# let all sigma_i = 1. 
sigma = numeric(n) + 1

### Step 2 - generate data
z = rnorm(n = n, mean = theta, sd = sigma)

### Step 3 - Soft thresholding for multiple lambda values
# setup multiple lambda values
lambda = 0:5

# Export to pdf
pdf('ex05_1b_step3.pdf',height=4,width=6)
plot(0,0,type='n',xlim = range(theta)*1.1, ylim = range(theta)*1.2, xlab=expression(theta),ylab=expression(paste(hat(theta),'(y)',sep='')),main='Soft Thresholding')
for(i in 1:length(lambda)){
  points = S_lambda(y = z, lambda = lambda[i]*sigma)
  points((theta), points, col = i, pch=19, cex = .5)
}
abline(0,1,lty=3)
text(22,23,expression(paste(hat(theta),'=',theta)))
legend('topleft',legend=lambda,title=expression(paste(lambda,' value')),col=1:6,pch=19)
dev.off()

### Step 4 -------------------------------------------------------------------------------
n = 100 # Number of datapoints
lambda = seq(0,3,length.out = 100) # arbitrary choice 
density = c(.01,.05,.10,.25,.50) # arbitrary choice


# Send to pdf - closes on dev.off()
pdf('ex05_1b_step4.pdf',height=6,width=8)
plot(0,0,type='n',xlim = range(lambda)*1.1, ylim = c(0,6), xlab=expression(lambda),ylab='Scaled MSE')

for(i in 1:length(density)){

  # initialize empty MSE vector
  MSE = lambda * 0
  
  # Sparse theta vector
  n.nonzero = round(n*density[i])
  theta = numeric(n)
  theta[sample(1:n,n.nonzero)] <- rpois(n.nonzero,4.5)
  
  # *known* vector of sigma's
  sigma = numeric(n) + 1
  
  # generate data
  z = rnorm(n = n, mean = theta, sd = sigma)
  
  # loop over lambda's
  for(j in 1:length(lambda)){
    # soft thresholding
    theta.hat = S_lambda(y = z, lambda = lambda[j]*sigma)
    
    #calculate MSE
    MSE[j] = mean((theta.hat-theta)^2)
  }
  
  # Find the minimum. 
  minimum = which.min(MSE)
  
  #Scale by minimum
  lines(lambda, MSE/MSE[minimum], col = i )
  
  # Dot the minimum. 
  points(lambda[minimum],1,col=i,pch=19,cex=2)
}

# Add legend
legend('bottomright',legend=1-density,title='Sparsity',col=1:length(density),pch=19)

# EXPORT to .pdf
dev.off()



###
### 2A ----------------------------------------------------------------------------------------
###
rm(list=ls())
library(glmnet)

# Load X into matrix
X = read.csv('diabetesX.csv')
X = as.matrix(X)

# Load Y into vector
Y = read.csv('diabetesY.csv',header=F)
Y = Y[,1]

# Choice of lambda's
n.lambda = 100
lambda = exp(seq(4,-4,length.out=n.lambda))

# model all 
model = glmnet(X,Y,alpha=1,lambda=lambda)

### Plot both coefficient sizes and MSE change
pdf('ex05_2a.pdf',height=6,width=6)
par(mfrow=c(2,1))

# plot beta's by lambda - yes R has a default, but I want the 
#  axes of the two plots to match 
plot(0,0,type='n',
     ylim=range(model$beta), ylab=expression(hat(beta)[lambda]),
     xlim=log(range(model$lambda)), xlab = expression(paste('log(',lambda,')')),
     main = 'Shrinkage of Coefficients'
     )
for(i in 1:nrow(model$beta)){
  lines(log(lambda),model$beta[i,],col=i)
}

# Plot in-sample MSE by lambda
MSE = model$lambda * 0

for(i in 1:ncol(model$beta)){
  MSE[i] = mean((Y - model$a0[i] - X%*%model$beta[,i])^2)
}
plot(log(model$lambda),MSE,type = 'l',
    main = 'In-sample MSE',
    ylab= "MSE",
    xlab = expression(paste('log(',lambda,')'))
)

# Export pdf
dev.off()


###
### 2B Cross validation ----------------------------------------------------
###

### I choose Leave-one-out cross validation (LOOCV)

# Data dimensions
nn = length(Y)
pp = ncol(X)
if(nn != nrow(X)){stop('Warning: data dimensions are not the same for X,Y')}

#
errors = matrix(0,nrow=nn,ncol=n.lambda)
for(i in 1:nn){
  model = glmnet(X[-i,],Y[-i],alpha=1,lambda=lambda)
  errors[i,] = Y[i] - predict(model,X[i,,drop=F])
}

# Get Mean OOS square error (MOOSE)
squared.errors = errors^2
MOOSE = apply(squared.errors,2,mean)

# Plot - export as PDF
pdf('ex05_2b.pdf',height=4,width=6)
plot(log(lambda),MOOSE,type='l',xlab=expression(paste("log(",lambda,")")))
dev.off()

# How many parameters at best lambda?
model = glmnet(X,Y,alpha=1,lambda=lambda)
sum(model$beta[,which.min(MOOSE)] != 0) #number nonzero betas for min MOOSE

###
### 2C Mallow's CP stat ----------------------------------------------------
###


# Data dimensions
nn = length(Y)
pp = ncol(X)
if(nn != nrow(X)){stop('Warning: data dimensions are not the same for X,Y')}

# Setup sequence of lambdas
n.lambda = 100
lambda = exp(seq(4,-4,length.out=n.lambda))

# initialize CP vector
CP = lambda * 0

# Run model with several lambda's
model = glmnet(X,Y,alpha=1,lambda=lambda)

# calculate MSE for each labmda
for(i in 1:ncol(model$beta)){
  errors = Y - model$a0[i] - X%*%model$beta[,i]
  CP[i] = mean((errors)^2) + 2*model$df[i]*var(errors)/nn
}

# Plot - export as PDF
pdf('ex05_2c.pdf',height=4,width=6)
plot(log(lambda),MSE,type='l',xlab=expression(paste("log(",lambda,")")),ylab='Error')
lines(log(lambda),MOOSE,col=2)
lines(log(lambda),CP,col=3)
legend('topleft',legend=c('IS MSE','MOOSE','CP'),col=1:3,lty=1)
dev.off()

# How many parameters at best lambda?
sum(model$beta[,which.min(CP)] != 0)
