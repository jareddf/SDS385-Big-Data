# Functions needed for Exercises 02, SDS 385 Big Data
# 2016.09.19

stochastic.gradient.descent = function(convex.function, gradient.function, starting.values, data, stepsize.function, function.smoothing.factor=.5, max.iter=100, min.iter=1, tol=.001, Polyak.Ruppert.burn, ...){
  #
  # Optimizes over beta vector, input to convex.function
  #
  pp = length(starting.values) 
  nn = nrow(data)
  
  # needed data structures. Returns "output" dataframe that includes each beta value, step size, etc. 
  number.details = 4
  output = data.frame(matrix(0, ncol = pp + number.details, nrow = max.iter + 1))
  beta = matrix(0, ncol = pp, nrow = max.iter + 1)
  names(output)[1:5] = c('Step','StepSize','RandomChoice','FunctionValue','Beta0')
  
  # Starting value. Defaults as 0's
  beta[1,] <- output[1, (1:pp) + number.details] <- starting.values
  
  # Initial values
  output$FunctionValue[1] = convex.function(beta[1,],data[sample(1:nn,1),,drop=F],previous = NA,function.smoothing.factor = function.smoothing.factor)
  output$StepSize = NA
  output$RandomChoice = NA
  output$Step = 0:max.iter

  if(!missing(Polyak.Ruppert.burn)){min.iter = max(min.iter,Polyak.Ruppert.burn)}
  
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
    
    # Polyak-Ruppert averaging of betas. 
    if(missing(Polyak.Ruppert.burn)){
		output[i+1,(1:pp)+number.details] <- beta[i+1,]
    } 
    else {
    		output[i+1,(1:pp)+number.details] <- (1/(i+1)) * beta[i+1,]  +  (i/(i+1)) * output[i,(1:pp)+number.details]
    }
    
    # Measure target function (Log Likeihood) of new beta - using only given point (and previous, for ). Requires averaged target function
    output$FunctionValue[i+1] = convex.function(
      beta = as.numeric(output[i+1,(1:pp)+number.details]), 
      data = data[choice,,drop=F], 
      previous = output$FunctionValue[i], 
      function.smoothing.factor = function.smoothing.factor
      )
    
    # Stoping criteria
#    if( (norm(beta[i,,drop=F]-beta[i+1,,drop=F]) < tol) & (i >= min.iter) ){
	if( (abs(output$FunctionValue[i] - output$FunctionValue[i+1])) < tol & (i >= min.iter)){
      return(list(betahat = beta[i+1,], output=output[1:(i+1),]))
    }
  }
  print('Failed to converge')
  return(list(output=output))
}

wrapper.logreg.negative.gradient = function(beta, data){
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

wrapper.logreg.likelihood = function(beta, data, previous=NA, function.smoothing.factor=NA){
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

wrapper.logreg.likelihood.EMA = function(beta, data, previous, function.smoothing.factor){
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
  
  # Exponential moving average, alpha = function.smoothing.factor. But l1 = l1
  if(is.na(previous)){
  	return(lik)
  }
  else{
  	return( function.smoothing.factor * lik + (1 - function.smoothing.factor) * previous )
  }
}


binomial.log.likelihood = function(y, w, m=1, adjustment=.00001){
  # Returns the binomial log likelihood, short the constant term. 
  # An adjustment is added within the logs to avoid numerical instability
  sum(y*log(w+adjustment) + (m-y)*log(1-w+adjustment))
}

negative.logistic.gradient = function(y, X, w, m=1){
  -t(X)%*%(y - w*m)
}

predict.logistic = function(X, beta){
  #equivalent to $w_i(\beta)$ in writeup
	1/(1+exp(-X%*%beta))
  # nn = dim(X)[1]
  # out = numeric(nn)
  # for(i in 1:nn){
  #   out[i] = 1/(1+exp(-X[i,]%*%beta))
  # }
  # return(out)
}

step.RobbinsMonro = function(t,t0=1,C,alpha){
	if(alpha > 1 | alpha < .5){stop('check step size "a" is in reasonable range')}
	if(C <= 0){stop('check step size "C" - currently not positive')}
	C*(t+t0)^(-alpha)
}

###
### 2c's simple example --------------------------------------------------------
###

# stochastic gradient descent to find mean of normal data ("simple")

simple.log.lik = function(beta, data, previous=NA, function.smoothing.factor=NA){
  # likelihood for normal data with known variance = 1.
  sum(dnorm(data, mean = beta[1], sd = 1, log = TRUE))
}

simple.log.lik.EMA = function(beta, data, previous, function.smoothing.factor){
  # Exponentially-weighted moving average likelihood for normal data with known variance = 1.
  lik = sum(dnorm(data, mean = beta[1], sd = 1, log = TRUE))

	# Exponentially weighted moving average. 
	if(is.na(previous)){
		return(lik)
	}
	else{
		return(function.smoothing.factor * lik + (1 - function.smoothing.factor) * previous)
	}
}

negative.simple.gradient = function(beta, data){
  # Negative gradient for normal data with known variance = 1.
  sum(beta - data)
}
