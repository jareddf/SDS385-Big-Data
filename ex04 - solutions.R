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
  ,stepsize.function = function(...){.01}
  ,data = all.data
  ,starting.values = numeric(ncol(X))
  ,function.smoothing.factor = .001
  ,max.iter=50000
  ,min.iter=1000
  ,tol= 1e-8
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
