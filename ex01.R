#Big Data HW1
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex01-functions.R')

###
### Problem 1c: Solving Weighted Least Squares with Matrix Decompositions
###

library(Matrix)
library(microbenchmark)

# checkout crossprod() function


# Generate data
NN = 1000
PP = c(5,10,20,50,100,200,500)
W = diag(NN)
med.in <- med.lu <- med.qr <- PP*0
# Benchmark
for(i in 1:length(PP)){
	X = cbind(1,matrix(rnorm(NN*(PP[i]-1)),nrow=NN,ncol=PP[i]-1))
	y = rnorm(NN,10,10) + X[,3]*4 
	bench = microbenchmark(wls.invert(y,X,W),wls.lu(y,X,W),wls.qr(y,X,W),times=10)
	med.in[i] = median(bench$time[bench$expr=='wls.invert(y, X, W)'])
	med.lu[i] = median(bench$time[bench$expr=='wls.lu(y, X, W)'])
	med.qr[i] = median(bench$time[bench$expr=='wls.qr(y, X, W)'])
}
plot(PP,med.in,type='b',pch=19,ylim=c(0,max(med.in,med.lu,med.qr)),xlab="Number of Parameters",ylab="Median Nanoseconds Required")
lines(PP,med.lu,type='b',pch=19,col=2)
lines(PP,med.qr,type='b',pch=19,col=3)
legend('topleft',col=1:3,lty=1,legend=c('Matrix Inversion','LU Decomposition','QR Decomposition'))
title(paste('N =',NN))




###
### Problem 1d 
###

#creates a 2x2 plot
par(mfrow=c(2,2))

# General parameters
NN = 1000 #size of generated dataset
PP = c(5,10,20,50,100,200,500) #varying number of randomly generated predictors
W = diag(NN) #Weights for weight least squares. In this case, equal-weighted

for(sparsity in c(.05,.1,.2,.5)){
  med.in <- med.lu <- med.qr <- med.sp <- med.glm <- PP*0
  for(i in 1:length(PP)){
    print(paste(sparsity, PP[i]))
    #Generate sparse data
    X = cbind(1,matrix(rnorm(NN*(PP[i]-1)),nrow=NN,ncol=PP[i]-1))
    mask = matrix(rbinom(NN*PP[i],1,sparsity), nrow=NN) # make sparse
    X = mask*X
    y = rnorm(NN,10,10) + X[,3]*4 
    #benchmark multiple methods
    bench = microbenchmark(wls.invert(y,X,W),wls.lu(y,X,W),wls.qr(y,X,W),wls.sp(y,X,W),glm(y~X-1,weights=diag(W)),times=10)
    med.in[i] = median(bench$time[bench$expr=='wls.invert(y, X, W)'])
    med.lu[i] = median(bench$time[bench$expr=='wls.lu(y, X, W)'])
    med.qr[i] = median(bench$time[bench$expr=='wls.qr(y, X, W)'])
    med.sp[i] = median(bench$time[bench$expr=='wls.sp(y, X, W)'])
    med.glm[i] = median(bench$time[bench$expr=='glm(y ~ X - 1, weights = diag(W))'])
  }
  plot(PP,med.in,type='b',pch=19,ylim=c(0,max(med.in,med.lu,med.qr)),xlab="Number of Parameters",ylab="Median Nanoseconds Required")
  lines(PP,med.lu,type='b',pch=19,col=2)
  lines(PP,med.qr,type='b',pch=19,col=3)
  lines(PP,med.sp,type='b',pch=19,col=4)
  lines(PP,med.glm,type='b',pch=19,col=5)
  legend('topleft',col=1:5,lty=1,legend=c('Matrix Inversion','LU Decomposition','QR Decomposition','Sparse Functions','glm() function'))
  title(paste('N = ',NN,' Sparsity = ',sparsity,sep=''))
}


###
### Problem 2B
###

data = read.csv('wdbc.csv',header=F)

#convert "y" to 0/1's. 
y = ifelse(data[,2] == 'M',1,0)
X = as.matrix(cbind(1,scale(data[,3:12])))

out = steepest.descent(y=y,X=X,max.iter=10000,tol=.001)

par(mfrow=c(1,2))
plot(out$output$Step,out$output$LogLikelihood,type='l',xlab='Step',ylab='Log Likelihood')
plot(out$output$Step,out$output$StepSize,type='l',xlab='Step',ylab='Step Size')
par(mfrow=c(1,1))
#
round(out$betahat,3)
round(glm(y~X-1,family=binomial)$coefficients,3)
(out$betahat - glm(y~X-1,family=binomial)$coefficients)/out$betahat

###
### Problem 2D
###

data = read.csv('wdbc.csv',header=F)
#convert "y" to 0/1's. 
y = ifelse(data[,2] == 'M',1,0)
X = as.matrix(cbind(1,scale(data[,3:12])))

starting.vector=numeric(11)
out = Newtons.method(y,X,starting.vector=numeric(11))
#
plot(out$output$Step,out$output$LogLikelihood,type='l',xlab='Step',ylab='Log Likelihood',main="Newton's Method with QR Decomposition")
#
library(xtable)
xtable(rbind( out$betahat,glm(y~X-1,family=binomial)$coefficients) ,digits=3)


