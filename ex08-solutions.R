# ex08 script
rm(list=ls())
setwd('~/Dropbox/School/SDS385_Big_Data/ex08-SpatialSmoothing/')
source('ex08-functions.R')
library(Matrix)

# Load data
dat = read.csv('~/Dropbox/School/SDS385_Big_Data/fmri_z.csv')
dat = Matrix(as.matrix(dat))


# Heatmap to display data
pdf('ex08_1c-heatmap.pdf')
image(dat,lwd=0,main='Raw Data')
dev.off()

#
D = makeD2_sparse(nrow(dat),ncol(dat))
DTD = t(D)%*%D
y = matrix(dat, ncol=1)
lambda = 1
A = DTD*lambda
diag(A) = diag(A) + 1


### Direct sovle with sparsity:
system.time(x.sparse <- solve(a = A,b = y))
#user  system elapsed 
#0.002   0.000   0.002 

# Plot
yhat = Matrix(as.vector(x.sparse), ncol=128, byrow = FALSE)
yhat[which(dat==0)] = 0
pdf('ex08_1c-sparsesolve.pdf')
image(yhat,lwd=0,main = "Direct Solver")
dev.off()



### JACOBI:
system.time(x.jacobi <- linearsolve.jacobi(A = A,b = y,max.iter=1000000,tol=.000001))
#user  system elapsed 
#0.694   0.121   0.821 

# Plot
yhat = Matrix(as.vector(x.jacobi), ncol=128, byrow = FALSE)
yhat[which(dat==0)] = 0
pdf('ex08_1c-jacobi.pdf')
image(yhat,lwd=0,main='Jacobi')
dev.off()

### Compare:
# Speed comparison
library(microbenchmark)
#microbenchmark(linearsolve.jacobi(A = A,b = y,max.iter=1000000,tol=.000001),solve(a = A,b = y))

# Small absolute differences between them? Note, this matches the tolerance of the iterative algorithm
mean(abs(x.jacobi - x.sparse) < .00001)



###
### Part 2 ----------------------------------------
###
system.time(x.tansey <- ADMM_graph.fused.lasso(y=y, D=D, lambda=lambda, max.iter=10000, tol = .00001))

yhat = Matrix(as.vector(x.tansey), ncol=128, byrow = FALSE)
yhat[which(dat==0)] = 0
pdf('ex08_2.pdf')
image(yhat,lwd=0,main = "Graph Fused Lasso")
dev.off()

# Comparison to original
difference = exp((-100:2)/10)
percent = difference * 0
for(i in 1:length(difference)){
  percent[i] = mean(abs(x.tansey - x.sparse) > difference[i])
}

# Plot of percent absolute differnce
pdf('ex08_2-threshold.pdf')
plot(log(difference), percent,type='l',
     main='Percent of Graph Differences Greater than Threshold',
     xlab = 'Log Threshold',
     ylab='Percent'
     )
abline(v = log(.0035),col=2)
dev.off()

# double check what's going on in above plot
hist(as.numeric(x.tansey - x.sparse),breaks=100)
plot(density(as.numeric(x.tansey - x.sparse)),main='Density of Differences')
plot(density(as.numeric(x.tansey - x.sparse),n=200000),xlim=c(-.007,0),main='Density of Differences - Zoomed')
