# This file contains the calculations and SOLUTIONS for exercises 03 for SDS 385 - Big Data. 
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex03 - functions.R')

###
### 1B ------------------------------------------------------------------------------------
###


# Data loading 
wdbc = read.csv('wdbc.csv',header=F)

# Convert M/B to binary
y = ifelse(wdbc[,2] == 'M',1,0)

# Take desired columns for X, and append to column of 1's for the intercept
X = as.matrix(cbind(1,scale(wdbc[,3:12])))

# Merge data into single object for use in modular functions. 
all.data = cbind(y,X)

# Set tolerance
tolerance = .001

# Set starting values
starting.values = numeric(ncol(X))


# Optimize
out = gradient.descent(
  convex.function = wrapper.logreg.negative.likelihood.EMA,
  gradient.function = wrapper.logreg.negative.gradient,
  stepsize.function = backtracking.line.search, 
  data = all.data,
  starting.values = starting.values,
  function.smoothing.factor=1, 
  max.iter=1000, 
  min.iter=20, 
  tol= tolerance,
  alpha0 = 1,
  c.constant = .2,
  rho.contraction.factor = .5
)
    

# Constant step size
out.constant = gradient.descent(
  convex.function = wrapper.logreg.negative.likelihood.EMA,
  gradient.function = wrapper.logreg.negative.gradient,
  stepsize.function = function(...){.01}, 
  data = all.data,
  starting.values = starting.values,
  function.smoothing.factor=1, 
  max.iter=1000, 
  min.iter=1, 
  tol= tolerance
)

# Constant step size
out.constant2 = gradient.descent(
  convex.function = wrapper.logreg.negative.likelihood.EMA,
  gradient.function = wrapper.logreg.negative.gradient,
  stepsize.function = function(...){.001}, 
  data = all.data,
  starting.values = starting.values,
  function.smoothing.factor=1, 
  max.iter=1000, 
  min.iter=1, 
  tol = tolerance
)

## Plot function over steps, and stepsize

# Export to pdf
pdf(file = 'ex03_1b.pdf',height = 4, width = 8)

# Two plots on 1 pdf
par(mfrow = c(1,2))

# Plot Step size over time
plot(log(out$output$Step),out$output$StepSize,
     type='l',
     xlab='Ln(Step)',
     ylab='Step Size'
)

# Plot convergence 
plot(log(out$output$Step),log(out$output$FunctionValue),
     type='l',
     xlab='Ln(Step)',
     ylab='Ln(Negative Log Likelihood)',
     #ylim = c(0,400),
     xlim = c(0,log(max(c(out.constant$output$Step,out.constant2$output$Step,out$output$Step))))
)
abline(h = wrapper.logreg.negative.likelihood.EMA(glm(y~X-1,family='binomial')$coefficients,all.data), col='grey')

lines(log(out.constant$output$Step),log(out.constant$output$FunctionValue), 
      col = 2)
lines(log(out.constant2$output$Step),log(out.constant2$output$FunctionValue), 
      col = 3)

# Legend
legend('topright',legend=c('Line Search','Constant = .01','Constant = .001'), col = 1:3, lty=1)

dev.off()

###
### 2b --------------------------------------------------------------------
###
# Quasi Newton's Method

# Clear memory for a fresh start
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex03 - functions.R')
source('ex02-functions.R')
source('ex01-functions.R')

# Data loading 
wdbc = read.csv('wdbc.csv',header=F)

# Convert M/B to binary
y = ifelse(wdbc[,2] == 'M',1,0)

# Take desired columns for X, and append to column of 1's for the intercept
X = as.matrix(cbind(1,scale(wdbc[,3:12])))

# Merge data into single object for use in modular functions. 
all.data = cbind(y,X)

# Set tolerance
tolerance = .001

# Set starting values
starting.values = numeric(ncol(X))


### Optimization routines

# Quasi Newton's method with BFGS update, line search 
out.bfgs = quasi.newtons.method.BFGS(
  convex.function = wrapper.logreg.negative.likelihood.EMA,
  gradient.function = wrapper.logreg.negative.gradient,
  stepsize.function = backtracking.line.search, 
  data = all.data,
  starting.values = starting.values,
  max.iter=1000, 
  min.iter=20, 
  tol= tolerance,
  alpha0 = .1,
  c.constant = .5,
  rho.contraction.factor = .9
)
print(log(dim(out.bfgs$output)))

# Newton's Method
out.newton = Newtons.method(y,X,starting.vector=numeric(11),tol = tolerance)

# Gradient Descent 
out.descent = steepest.descent(y,X,startbeta=numeric(11),m=1,max.iter=1000,tol=tolerance,save.all=TRUE)

# Gradient Descent with Line search
out.linesearch = gradient.descent(
    convex.function = wrapper.logreg.negative.likelihood.EMA,
    gradient.function = wrapper.logreg.negative.gradient,
    stepsize.function = backtracking.line.search, 
    data = all.data,
    starting.values = starting.values,
    function.smoothing.factor=1, 
    max.iter=1000, 
    min.iter=20, 
    tol= tolerance,
    alpha0 = 1,
    c.constant = .2,
    rho.contraction.factor = .5
  )

# Stochastic Gradient Descent with constant step size
out.stochastic = stochastic.gradient.descent(
  convex.function = wrapper.logreg.negative.likelihood.EMA,
  gradient.function = wrapper.logreg.negative.gradient,
  starting.values = starting.values,
  data = all.data,
  stepsize.function = function(...){.01},
  function.smoothing.factor = .1,
  max.iter=10000,min.iter = 100, tol=tolerance)


### Export to PDF
pdf(file = 'ex03_2b.pdf',height = 8, width = 16)

# Two plots on 1 pdf
par(mfrow = c(1,2))

# Plot Step size over time
plot(log(out.bfgs$output$Step+1),out.bfgs$output$StepSize,
     type='l',
     xlab='Ln(Step)',
     ylab='Step Size'
)

# Plot convergence 
plot(log(out.bfgs$output$Step+1),log(out.bfgs$output$FunctionValue)
     ,type='l'
     ,xlab='Ln(Step)'
     ,ylab='Ln(Negative Log Likelihood)'
     ,ylim = c(4,6)
     ,xlim = c(0,6)
     ,lwd = 3
)
lines(log(out.newton$output$Step+1),log(-(out.newton$output$LogLikelihood)), lty=3, lwd=2, col = 2)
lines(log(out.descent$output$Step+1),log(-(out.descent$output$LogLikelihood)), lty=3, lwd=2, col = 3)
lines(log(out.linesearch$output$Step+1),log(out.linesearch$output$FunctionValue), lty=3, lwd=2, col = 4)
lines(log(out.stochastic$output$Step+1),log(nrow(all.data)*out.stochastic$output$FunctionValue), lty=3, lwd=2, col = 5)

optimization.names = c('BFGS',"Newton's Method",'Gradient Descent','GD w/ Linesearch','Stochastic GD')
legend('bottomleft',legend=optimization.names,col=1:5,lty=c(1,3,3,3,3),lwd=3)

dev.off()
