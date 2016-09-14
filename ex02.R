rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/')
source('ex02-functions.R')

###
### 2c -------------------------------------------------------------------------------------
###

### Simulated Dataset - Normal distributed data with known variance=1
source('ex01-functions.R')

b = 3
y = cbind(rnorm(10000,mean=b,sd=1))


# Plot different step sizes
sizes = c(.001,.01,.1)
par(mfrow=c(1,length(sizes)))
for(s in sizes){
  constant.step=function(iteration){s}
  out = stochastic.gradient.descent(
    convex.function = simple.log.lik,
    gradient.function = negative.simple.gradient,
    starting.values = 0,
    data = y,
    stepsize.function = constant.step,
    max.iter=10000,min.iter = 1, tol=.00001)
  
  plot(out$output$Step,out$output$Beta0,type='l',xlab='Step',ylab='Parameter Estimate',main=paste('Step size =',constant.step(1)),ylim=c(0,4))
  abline(h=mean(y), col=2)
}




### Real dataset

wdbc = read.csv('wdbc.csv',header=F)
y = ifelse(wdbc[,2] == 'M',1,0)
X = as.matrix(cbind(1,scale(wdbc[,3:12])))
all.data = cbind(y,X)


constant.step=function(iteration){
  .01
}

out = stochastic.gradient.descent(
  convex.function = wrapper.logreg.likelihood,
  gradient.function = wrapper.logreg.negative.gradient,
  starting.values = numeric(ncol(X)),
  data = all.data,
  stepsize.function = constant.step,
  max.iter=1000,tol=.000001)
  
out$output[,1:5]
plot(out$output$Step,out$output$FunctionValue,type='l',xlab='Step',ylab='Log Likelihood',main="Stochastic Gradient Descent")
