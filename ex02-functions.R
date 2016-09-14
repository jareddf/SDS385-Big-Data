
stochastic.gradient.descent = function(convex.function,gradient.function,starting.values,data,stepsize.function,max.iter=100,min.iter=1,tol=.001,...){
  #
  # Optimizes over beta vector, input to convex.function
  #
  pp = length(starting.values) 
  nn = nrow(data)
  
  # needed data structures. 
  output = data.frame(matrix(0,ncol=pp+4,nrow=max.iter+1))
  beta = matrix(0,ncol=pp,nrow=max.iter+1)
  names(output)[1:5] = c('Step','StepSize','RandomChoice','FunctionValue','Beta0')
  
  # Starting value. Defaults as 0's
  beta[1,]=starting.values
  
  # Initial values
  output$FunctionValue[1] = convex.function(beta[1,],data)
  output$StepSize = NA
  output$RandomChoice = NA
  output$Step = 0:max.iter
  
  
  # Loop
  for(i in 1:max.iter){
    # Stochastic piece: randomly choose single datapoint
    choice = sample(1:nn,1)
    output$RandomChoice[i+1] = choice
    
    # Find gradient at current beta
    gradient = gradient.function(beta = beta[i,], data = data[choice,,drop=F])
    
    # Determine stepsize
    output$StepSize[i+1] = stepsize.function(i,...)
    
    # Step down the hill	
    beta[i+1,] = beta[i,] - output$StepSize[i+1] * gradient 
    
    # Measure Log Likeihood of new beta
    output$FunctionValue[i+1] = convex.function(beta=beta[i+1,],data)
    
    # Stoping criteria
    if( (norm(beta[i,,drop=F]-beta[i+1,,drop=F]) < tol) & (i >= min.iter) ){
      output[,(1:pp)+4] <- beta
      return(list(betahat = beta[i+1,], output=output[1:(i+1),]))
    }
  }
  output[,(1:pp)+4] <- beta
  print('Failed to converge')
  return(list(output=output))
}

wrapper.logreg.negative.gradient = function(beta,data){
  # function for use within optimization (minimization) function. 
  # Input: beta vector, data matrix where first column is response variable Y
  # Output: negative gradient of log binomial likelihood for logistic regression
  
  # Strip Y vector off input data matrix
  X = data[,2:ncol(data),drop=F]
  y = data[,1,drop=F]
  
  # Get predict "success" probabilities
  w.hat = predict.logistic(X=X, beta=beta)
  
  # Compute and return negative gradient
  gradient = negative.logistic.gradient(y=y, X=X, w=w.hat, m=1)
  return(gradient)
}

wrapper.logreg.likelihood = function(beta,data){
  # function for use within optimization function. 
  # Input: beta vector, data matrix where first column is response variable Y
  # Output: log binomial likelihood for logistic regression
  
  # Strip Y vector off input data matrix
  X = data[,2:ncol(data),drop=F]
  Y = data[,1,drop=F]
  
  # Get predict "success" probabilities
  w.hat = predict.logistic(X=X, beta=beta)
  
  # Calculate and return likelihood 
  lik = binomial.log.likelihood(y=Y, w=w.hat)
  return(lik)
}

binomial.log.likelihood = function(y, w, m=1, adjustment=.00001){
  # Returns the binomial log likelihood, short the constant term. 
  # An adjustment is added within the logs to avoid numerical instability
  sum(y*log(w+adjustment) + (m-y)*log(1-w+adjustment))
}

negative.logistic.gradient = function(y,X,w,m=1){
  -t(X)%*%(y - w*m)
}

predict.logistic = function(X,beta){
  #equivalent to $w_i(\beta)$ in writeup
  nn = dim(X)[1]
  out = numeric(nn)
  for(i in 1:nn){
    out[i] = 1/(1+exp(-X[i,]%*%beta))
  }
  return(out)
}


###
#
#
simple.log.lik = function(beta,data){
  # likelihood for normal data with known variance = 1.
  sum(dnorm(data, mean = beta[1], sd = 1, log = TRUE))
}

negative.simple.gradient = function(beta,data){
  # Negative gradient for normal data with known variance = 1.
  sum(beta - data)
}
