rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex02-functions.R')

###
### 2c -------------------------------------------------------------------------------------
###

### Simulated Dataset - Normal distributed data with known variance=1. NOW with Exponentially moving averages (EMA) of log likelihood!!!
source('ex01-functions.R')

# Arbitrary choice of data mean 
b = 3

# Generate random N(b,1) data, cbind to ensure column vector. 
y = cbind(rnorm(10000,mean=b,sd=1))


## Plot different step sizes
sizes = c(.001,.01,.1)

# Creates multiple-pane plot. New plots proceed down each column. 
par(mfcol=c(2,length(sizes)))

# Loop through step sizes
for(s in sizes){
  
  # Set step size function to constant value s
  constant.step=function(iteration){s}
  
  # Run optimization
  out = stochastic.gradient.descent(
    convex.function = simple.log.lik.EMA,
    gradient.function = negative.simple.gradient,
    starting.values = 0,
    data = y,
    stepsize.function = constant.step,
    function.smoothing.factor = .1,
    max.iter=10000,min.iter = 100, tol=.00001)
  
  # For each step size, first plot log likelihood then mean estimate below it. Each with "true" value marked 
  plot(out$output$Step, out$output$FunctionValue, 
       type='l', 
       xlab='Step', 
       ylab='Log Likelihood', 
       main=paste('Step size =',constant.step(1)) 
       )
  abline(h=simple.log.lik(mean(y),y), col=2)
  plot(out$output$Step, out$output$Beta0,
       type='l',xlab='Step',
       ylab='Parameter Estimate',
       main=paste('Step size =',constant.step(1)),
       ylim=c(0,4)
       )
  abline(h=mean(y), col=2)
}



# !! Code used with different convergence criteria, before exponentially weighted moving average of log likelihood. 
### Real dataset -
# 
# wdbc = read.csv('wdbc.csv',header=F)
# y = ifelse(wdbc[,2] == 'M',1,0)
# X = as.matrix(cbind(1,scale(wdbc[,3:12])))
# all.data = cbind(y,X)
# 
# 
# # Plot different step sizes
# sizes = c(.001,.01,.1,1)
# par(mfrow=c(1,1))
# for(s in sizes){
#   constant.step=function(iteration,function.value){s}
#   out = stochastic.gradient.descent(
#     convex.function = wrapper.logreg.likelihood,
#     gradient.function = wrapper.logreg.negative.gradient,
#     starting.values = numeric(ncol(X)),
#     data = all.data,
#     stepsize.function = constant.step,
#     max.iter=10000,tol=.0000001)
#   if(s == sizes[1]){
#     plot(out$output$Step,out$output$FunctionValue,type='l',xlab='Step',ylab='Log Likelihood',main='Stochastic Gradient Descent - WDBC Data',ylim=c(-4,0))
#     abline(h= wrapper.logreg.likelihood(glm(y~X-1,family='binomial')$coefficients,all.data), col='grey')
#   }
#   else{
#     lines(out$output$Step,out$output$FunctionValue,col=which(sizes == s))
#   }
# }
# legend('bottomright',legend=sizes,col=1:length(sizes),lty=1,title='Step Size')






### Real dataset - with exponential moving average log likelihood

# Load data
wdbc = read.csv('wdbc.csv',header=F)

# Change binary response value to actual binary values. 
y = ifelse(wdbc[,2] == 'M',1,0)

# Select desired columns, with column of 1's for intercept term
X = as.matrix(cbind(1,scale(wdbc[,3:12])))

# Bundle both together as input into modularized functions within optimization
# I've made all function inputs in stochastic gradient descent function accept 
# a generic "data" varible as an input, so the code is general enough to be used again. 
all.data = cbind(y,X)


# Plot different step sizes
sizes = c(.001,.01,.1,1)

# side by side plots, each containing different sizes' results, but differing in alpha
par(mfrow=c(1,2))
for(a in c(.001,.01)){
  for(s in sizes){
    
    # Set constant step size. Must be function for input in stochastic.gradient.descent
    constant.step=function(iteration,function.value){s}
    
    # Optimize!
    out = stochastic.gradient.descent(
      convex.function = wrapper.logreg.likelihood.EMA,
      gradient.function = wrapper.logreg.negative.gradient,
      starting.values = numeric(ncol(X)),
      data = all.data,
      stepsize.function = constant.step,
      function.smoothing.factor = a,
      max.iter=10000,tol=1E-8)
    
    # Plot function value if first size, otherwise just add lines onto that plot. 
    if(s == sizes[1]){
      plot(out$output$Step,out$output$FunctionValue,type='l',xlab='Step',ylab='Log Likelihood',main=paste('WDBC Data - EMA alpha =',a),ylim=c(-1,0))
      abline(h= (1/length(y))* wrapper.logreg.likelihood(glm(y~X-1,family='binomial')$coefficients,all.data), col='grey')
    }
    else{
      lines(out$output$Step,out$output$FunctionValue,col=which(sizes == s))
    }
  }
}

# add simple legend. 
legend('bottomright',legend=sizes,col=1:length(sizes),lty=1,title='Step Size')



###
### 2D -----------------------------------------------------------------------------------------
###
# Use Robbins-Monro step sizes, on different values of C>0 and alpha in [0.5,1]

# Data loading and manipulation commands recopied from previous section, for convience of this section!
wdbc = read.csv('wdbc.csv',header=F)
y = ifelse(wdbc[,2] == 'M',1,0)
X = as.matrix(cbind(1,scale(wdbc[,3:12])))
all.data = cbind(y,X)

# side by side plots, each containing different C and alpha
pdf(file = 'ex02_2d.pdf')
par(mfrow=c(3,3))

# loop over different alpha
for(a in c(.85,.9,.95)){
  
  # loop over different C 
  for(C in c(1,2,3)){
    
    # Optimize!
    out = stochastic.gradient.descent(
      convex.function = wrapper.logreg.likelihood.EMA,
      gradient.function = wrapper.logreg.negative.gradient,
      starting.values = numeric(ncol(X)),
      data = all.data,
      stepsize.function = step.RobbinsMonro,
      function.smoothing.factor = .005,
      max.iter=10000,
      tol=1E-7,
      alpha = a,
      C = C)
    
    # Plot loglikelihood value at each step and glm's likelihood. 
      plot(out$output$Step,out$output$FunctionValue,
           type='l',
           xlab='Step',
           ylab='Log Likelihood',
           main=paste('alpha = ',a,', C = ',C,sep = ''),
           ylim = c(-1,0),
           xlim = c(0,10000)
           )
      abline(h= (1/length(y))* wrapper.logreg.likelihood(glm(y~X-1,family='binomial')$coefficients,all.data), col='grey')
  }
}
dev.off()


###
### 2E -----------------------------------------------------------------------------------------
###
# Now also use Polyak-Ruppert averaging

# Data loading and manipulation commands recopied from previous section, for convience of this section!
wdbc = read.csv('wdbc.csv',header=F)
y = ifelse(wdbc[,2] == 'M',1,0)
X = as.matrix(cbind(1,scale(wdbc[,3:12])))
all.data = cbind(y,X)

# side by side plots, each containing different C and alpha
pdf(file = 'ex02_2e.pdf')
par(mfrow=c(3,3))

# loop over different alpha
for(a in c(.7,.9,.95)){
  
  # loop over different C 
  for(C in c(.1,1,5)){
    
    # Optimize!
    out = stochastic.gradient.descent(
      convex.function = wrapper.logreg.likelihood.EMA,
      gradient.function = wrapper.logreg.negative.gradient,
      starting.values = numeric(ncol(X)),
      data = all.data,
      stepsize.function = step.RobbinsMonro,
      function.smoothing.factor = .005,
      max.iter=10000,
      tol=1E-7,
      Polyak.Ruppert.burn = 1000,
      alpha = a,
      C = C
      )
    
    # Plot loglikelihood value at each step and glm's likelihood. 
    plot(out$output$Step,out$output$FunctionValue,
         type='l',
         xlab='Step',
         ylab='Log Likelihood',
         main=paste('alpha = ',a,', C = ',C,sep = ''),
         ylim = c(-1,0),
         xlim = c(0,10000)
    )
    abline(h= (1/length(y))* wrapper.logreg.likelihood(glm(y~X-1,family='binomial')$coefficients,all.data), col='grey')
  }
}
dev.off()

