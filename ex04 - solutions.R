# This file contains the calculations and SOLUTIONS for exercises 04 for SDS 385 - Big Data. 
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex04 - functions.R')

###
### 1A ------------------------------------------------------------------------------------
###


# Data loading 
wdbc = read.csv('wdbc.csv',header=F)

# Convert M/B to binary
y = ifelse(wdbc[,2] == 'M',1,0)

# Take desired columns for X, and append to column of 1's for the intercept
X = as.matrix(cbind(1,scale(wdbc[,3:12])))

# Merge data into single object for use in modular functions. 
all.data = cbind(y,X)

# Set starting values
starting.values = numeric(ncol(X))

# Optimize
out = gradient.descent(
  convex.function = wrapper.logreg.negative.likelihood.EMA,
  gradient.function = wrapper.logreg.negative.gradient,
  stepsize.function = backtracking.line.search, 
  data = all.data,
  starting.values = starting.values,
  function.smoothing.factor=.01, 
  max.iter=10000, 
  min.iter=1, 
  tol= .00001,
  alpha0 = .1,
  c.constant = .2,
  rho.contraction.factor = .5,
  stochastic = T,
  Polyak.Ruppert.burn = 500,
  minibatch.size = 50,
  minibatch.frequency = 50
)

### export 1x2 plot to pdf
pdf(file = 'ex04_1a.pdf',width=10,height=6)
par(mfrow=c(1,2))

# Plot step sizes over iterations. 
plot(out$output$Step,out$output$StepSize
     ,type='l'
     ,ylab = 'Step Size'
     ,xlab = 'Step'
)

# Plot likelihood over iterations. 
plot(out$output$Step,nrow(all.data)*out$output$FunctionValue
     ,type='l'
     ,ylab = 'Negative Log Likelihood'
     ,xlab = 'Step'
     )

# include glm's likelihood value for comparison. 
glm.lik = wrapper.logreg.negative.likelihood.EMA(beta = glm(y~X-1,family='binomial')$coefficients, data = all.data)
abline(h = glm.lik, col=2)
text(25,85,'glm()' ,col=2)

#export
dev.off()




###
### 1b -----------------------------------------------------
###

# ### James' Example 
# 
# x = seq(0,4*pi, length=1000)
# 
# # Slightly noisy cosine curve
# sigma = 0.025
# y = cos(x) + rnorm(1000, 0, sigma)
# 
# # Function is easy to see despite the noise
# plot(x, y, pch=19, col='grey')
# curve(cos(x), add=TRUE, col='red', lwd=2)
# 
# # First differences are super noisy despite small sigma
# plot(tail(x,-1), diff(y)/diff(x), pch=19, col='grey')
# curve(-sin(x), add=TRUE, col='red', lwd=2) # true first derivative
# 
# # Second differences? Yikes!
# plot(tail(x,-2), diff(diff(y))/diff(diff(x)))


### Fresh start. 
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex04 - functions.R')

# Data loading 
wdbc = read.csv('wdbc.csv',header=F)
y = ifelse(wdbc[,2] == 'M',1,0) # Convert M/B to binary
X = as.matrix(cbind(1,scale(wdbc[,3:12]))) # Take desired columns for X, and append 1's
all.data = cbind(y,X) # Merge data into single object for use in modular functions. 


out = adagrad(
  convex.function = wrapper.logreg.negative.likelihood.EMA
  ,gradient.function = wrapper.logreg.negative.gradient
  ,stepsize.function = function(...){.01}#backtracking.line.search
  ,data = all.data
  ,starting.values = numeric(ncol(X))
  ,function.smoothing.factor = .01
#  ,alpha0 = .01
#  ,c.constant = .2
#  ,rho.contraction.factor = .5
  ,max.iter=5000
  ,min.iter=1
  ,tol= 1e-7
  ,Polyak.Ruppert.burn = 1000
)


### export to pdf
pdf(file = 'ex04_1b.pdf',width=10,height=6)



# Plot likelihood over iterations. 
plot(out$output$Step,nrow(all.data)*out$output$FunctionValue
     ,type='l'
     ,ylab = 'Negative Log Likelihood'
     ,xlab = 'Step'
     ,ylim = c(0,400)
)

# include glm's likelihood value for comparison. 
glm.lik = wrapper.logreg.negative.likelihood.EMA(beta = glm(y~X-1,family='binomial')$coefficients, data = all.data)
abline(h = glm.lik, col=2)
text(150,85,'glm()' ,col=2)

# export
dev.off()












###
### 2 -----------------------------------------------------
###

# currently only fixed step size. 
# no Polyak Ruppert averaging of the betas. 


### Fresh start. 
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex04 - functions.R')

# Data loading 
wdbc = read.csv('wdbc.csv',header=F)
y = ifelse(wdbc[,2] == 'M',1,0) # Convert M/B to binary
X = as.matrix(cbind(1,scale(wdbc[,3:12]))) # Take desired columns for X, and append 1's
all.data = cbind(y,X) # Merge data into single object for use in modular functions. 


out = fast.sgd(
  convex.function = wrapper.logreg.negative.likelihood.EMA
  ,gradient.function = wrapper.logreg.negative.gradient
  ,stepsize.function = function(...){1}
  ,data = all.data
  ,starting.values = numeric(ncol(X))
  ,function.smoothing.factor = .001
  ,max.iter=20000
  ,min.iter=20000
  ,tol= 1e-6
  #,lik.penalty=0.001
)


### export to pdf
#pdf(file = 'ex04_2.pdf',width=10,height=6)



# Plot likelihood over iterations. 
plot(out$output$Step,nrow(all.data)*out$output$FunctionValue
     ,type='l'
     ,ylab = 'Negative Log Likelihood'
     ,xlab = 'Step'
     ,ylim = c(0,400)
)

# include glm's likelihood value for comparison. 
glm.lik = wrapper.logreg.negative.likelihood.EMA(beta = glm(y~X-1,family='binomial')$coefficients, data = all.data)
abline(h = glm.lik, col=2)
text(150,85,'glm()' ,col=2)

# export
#dev.off()


###
### C++ ---------------------------------------------------------------------
###


### Fresh start. 
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
library(Matrix)
library(Rcpp)
source('ex04 - functions.R')
sourceCpp('ex04-SGD-denseX.cpp')

# Data loading 
wdbc = read.csv('wdbc.csv',header=F)
# Convert M/B to binary
y = ifelse(wdbc[,2] == 'M',1,0) 
# Take desired columns for X, and append 1's, and transpose
X = as.matrix(cbind(1,scale(wdbc[,3:12])))
# Merge data into single object for use in modular functions.
all.data = cbind(y,X)  


out = sgd_logit_fast(t(X),y,m=rep(1,length(y)), starting_values = numeric(nrow(X)), stepsize = 1, iterations = 20000, likelihood_smoothing_factor = .001)
plot(out$nll*nrow(X),type='l',col=3)


# include glm's likelihood value for comparison. 
glm.lik = wrapper.logreg.negative.likelihood.EMA(beta = glm(y~X-1,family='binomial')$coefficients, data = all.data)
abline(h = glm.lik, col=2)
text(150,85,'glm()' ,col=2)

### Speed test
library(microbenchmark)
RRR = function(){
  out = fast.sgd(
    convex.function = wrapper.logreg.negative.likelihood.EMA
    ,gradient.function = wrapper.logreg.negative.gradient
    ,stepsize.function = function(...){1}
    ,data = all.data
    ,starting.values = numeric(ncol(X))
    ,function.smoothing.factor = .001
    ,max.iter=20000
    ,min.iter=20000
    ,tol= 1e-6
    #,lik.penalty=0.001
  )
}
CCC = function(){
  out = sgd_logit_dense(t(X),y,m=rep(1,length(y)), starting_values = numeric(nrow(X)), stepsize = 1, iterations = 20000, likelihood_smoothing_factor = .001)
}

microbenchmark(RRR(),CCC(),times=10)


###
### SPARSE X in c++ on "Big" data ----------------------------------------------------------
###

### from James' code.
### Fresh start (similar as above, but can run just this subsection of the script)
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
library(Matrix)
library(Rcpp)

# Compile Rcpp functions
sourceCpp('james_sgdlogit.cpp')
sourceCpp('ex04-SGD-sparseX.cpp')

# Read in serialized objects
system.time(X <- readRDS('~/Documents/DATA/url_X.rds'))
system.time(y <- readRDS('~/Documents/DATA/url_y.rds'))

#Scale X without destroying sparsity???

# Obtain data dimensions. 
n = length(y)
p = ncol(X)

# column-oriented storage of each observation
system.time(tX <- t(X))

# Start beta vector as zeros
init_beta = rep(0.0, p)

# James' SGD (from his posted R script)
system.time(sgd_james <- sparsesgd_logit(tX, y, rep(1,n), eta = 2, npass=1, beta0 = init_beta, lambda=1e-8, discount = 0.001))

# My SGD with same optimization parameters
system.time(sgd_jared <- sgd_logit_sparse(tX, y, rep(1,n), stepsize = 2, npass=1, starting_values = init_beta, likelihood_penalty = 1e-8, smooth = 0.001))

# Plot my SGD's likelihood vs James'. Export as pdf. 
pdf('ex04_2.pdf',height = 3, width = 6)
length.out = length(sgd_james$nll_tracker)
subset = seq(1,length.out,1000)
plot(sgd_james$nll_tracker[subset],type='l',ylab='Negative Likelihood')
lines(sgd_jared$nll[subset],col=2,lty=3)
legend('topright',legend=c("James's .cpp function","Jared's .cpp function"),col=1:2,lty=c(1,3))
dev.off()


###
### Crossvalidate/Train/Test
###
sourceCpp('ex04-SGD-sparseX.cpp')

# Here I seperate to training/test data to select a best likelihood penalty. 
train.percentage = 0.7
train.size = round(n * train.percentage)
train.indices = sample(1:n,train.size)

# loop over different l2 penalties
lambda = exp(-15:5)#c(1e-14,1e-10,1e-4,1e-2,1e-1,.99,1,2,)
predictive.accuracy <- sensitivity <- specificity <- lambda*0 #faster than numeric(length(lambda))

for(i in 1:length(lambda)){
  
  # fit logistic regression using sparse sgd with l2 penalty
  system.time(sgd_jared <- sgd_logit_sparse(tX[,train.indices], y[train.indices], rep(1,train.size), stepsize = 1, npass=2, starting_values = init_beta, likelihood_penalty = lambda[i], smooth = 0.001))
  #system.time(sgd_james <- sparsesgd_logit(tX[,train.indices], y[train.indices], rep(1,train.size), eta = 1, npass=2, beta0 = init_beta, lambda=lambda[i], discount = 0.001))
  
  # plot target function to check
  length.out = length(sgd_jared$nll)
  subset = seq(1,length.out,100)
  plot(subset,sgd_jared$nll[subset],type='l',ylab='Negative Log Likelihood')
  abline(v=train.size,col=2)
  
  # predict 0 or 1
  xbeta = sgd_jared$intercept + X[-train.indices,]%*%sgd_jared$beta
  predicted = 1/(1+exp(-xbeta))
  predictions = ifelse(predicted>.5,1,0)
  
  # record accuracy rate
  predictive.accuracy[i] =  mean(predictions==y[-train.indices])
  sensitivity[i] = sum((predictions == 1) & (y[-train.indices] == 1))/sum(y[-train.indices]==1)
  specificity[i] = sum(predictions == 0 & y[-train.indices] == 0)/sum(y[-train.indices]==0)
}

#plot predictive performance for differing lambda
pdf('ex04_2b.pdf',height = 4, width = 6)
plot(log(lambda),predictive.accuracy,type='l',ylim=c(.5,1),lwd=2,ylab='Percentage Correct')
lines(log(lambda),sensitivity,col=2,lwd=2)
lines(log(lambda),specificity,col=3,lwd=2)
legend('bottomleft',legend=c('Overall Accuracy','Sensitivity','Specificity'),lty=1,lwd=2,col=1:3)
dev.off()
