# This file contains the functions needed to complete exercises 03 for SDS 385 - Big Data. 

###
### Line Search Section  -----------------------------------------------------------------
###

gradient.descent = function(convex.function, gradient.function, stepsize.function, data, starting.values, function.smoothing.factor=1, max.iter=1000, min.iter=1, tol=.001, stochastic = F, Polyak.Ruppert.burn, ...){
  #
  # Optimizes over beta vector, input to convex.function
  #
  
  # Dimensions
  pp = length(starting.values) 
  nn = nrow(data)
  
  # Needed data structures. Returns "output" dataframe that includes each beta value, step size, etc. 
  number.details = 4
  beta = matrix(0, ncol = pp, nrow = max.iter + 1)
  output = data.frame(matrix(0, ncol = pp + number.details, nrow = max.iter + 1))
  names(output)[1:5] = c('Step','StepSize','RandomChoice','FunctionValue','Beta0')
  
  # Starting value. Defaults as 0's
  beta[1,] <- output[1, (1:pp) + number.details] <- starting.values
  
  # Initial values
  output$StepSize = NA
  output$RandomChoice = NA
  output$Step = 0:max.iter
  choice = 1:nn
  if(stochastic){ choice = sample(1:nn,1)  }
  output$FunctionValue[1] = convex.function(beta[1,], data[choice,,drop=F], previous = NA, function.smoothing.factor = function.smoothing.factor)
  
  if(!missing(Polyak.Ruppert.burn)){min.iter = max(min.iter,Polyak.Ruppert.burn)}
  
  # Loop
  for(i in 1:max.iter){
    # Stochastic piece: randomly choose single datapoint
    if(stochastic){
      choice = sample(1:nn,1)
      output$RandomChoice[i+1] = choice
    }
    
    # Find gradient at current beta
    gradient = gradient.function(beta = beta[i,], data = data[choice,,drop=F])
    
    # Determine stepsize
    output$StepSize[i+1] = stepsize.function(iteration = i, 
                                             convex.function = convex.function, 
                                             direction.vector = -gradient, 
                                             current.gradient.vector = gradient, 
                                             current.beta.vector = beta[i,], 
                                             data = data[choice,,drop=F],
                                             ...)
    
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
    
    # Stopping criteria
    if( (abs(output$FunctionValue[i] - output$FunctionValue[i+1])) < tol & (i >= min.iter)){
      return(list(betahat = output[i+1,(1:pp)+number.details], output=output[1:(i+1),]))
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


wrapper.logreg.negative.likelihood.EMA = function(beta, data, previous = NA, function.smoothing.factor = 1){
  # wrapper of functions for use within optimization function. 
  # Input: beta vector, data matrix where first column is response variable Y
  # Output: log binomial likelihood for logistic regression
  # Default function.smoothing.factor = 1 means that there's no averaging by default. 
  
  # Strip Y vector off input data matrix
  X = data[,2:ncol(data),drop=F]
  Y = data[,1,drop=F]
  
  # Get predict "success" probabilities
  w.hat = predict.logistic(X=X, beta=beta)
  
  # Calculate and return negative likelihood 
  lik = -binomial.log.likelihood(y=Y, w=w.hat)
  
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
}

backtracking.line.search = function(iteration, convex.function, direction.vector, current.gradient.vector, current.beta.vector, data, alpha0 = 1, rho.contraction.factor = .9, c.constant = .5){
  # Returns step size (scalar) for optimization that fulfills sufficient decrease condition
  # Requires as input: convex.function, gradient.vector, direction.vector, 
  # current.beta, alpha0 > 0, rho.contraction.factor in (0,1), c.constant in (0,1)
  # input "iteration" is just there to match other step size functions ... 
  
  # Check for correct ranges of function parameters
  if(alpha0 <= 0){stop('alpha0 is not positive')}
  if(rho.contraction.factor <= 0 | rho.contraction.factor >= 1){stop('rho.contraction.factor is not in (0,1)')}
  if(c.constant <= 0 | c.constant >= 1){stop('c.constant is not in (0,1)')}
  
  # Set initial alpha value
  alpha = alpha0
  
  # Important values that need only be calculated once
  current.function.value = convex.function(current.beta.vector, data)
  dot.product = c.constant *  t(current.gradient.vector) %*% direction.vector
  
  # Adjust to correct appropriate step size (alpha)
  while( convex.function(beta = (current.beta.vector + alpha*direction.vector), data = data) > (current.function.value + alpha * dot.product) ){
    alpha = alpha * rho.contraction.factor
  }
  
  # Return better step size!
  return(alpha)
}
  
  

###
### Quasi Newton's Method ---------------------------------------------------------------
###


quasi.newtons.method.BFGS = function(convex.function, gradient.function, stepsize.function, data, starting.values, function.smoothing.factor=1, max.iter=1000, min.iter=1, tol=.001, stochastic = F, Polyak.Ruppert.burn, ...){
  #
  # Optimizes over beta vector, input to convex.function
  #
  
  # Dimensions
  pp = length(starting.values) 
  nn = nrow(data)
  
  # Needed data structures. Returns "output" dataframe that includes each beta value, step size, etc. 
  number.details = 4
  beta <- gradient <- matrix(0, ncol = pp, nrow = max.iter + 1)
  output = data.frame(matrix(0, ncol = pp + number.details, nrow = max.iter + 1))
  names(output)[1:5] = c('Step','StepSize','RandomChoice','FunctionValue','Beta0')
  
  
  # Starting value. 
  beta[1,] <- output[1, (1:pp) + number.details] <- starting.values
  
  # Initial values
  output$StepSize = NA
  output$RandomChoice = NA
  output$Step = 0:max.iter
  choice = 1:nn
  
  # Options
  if(stochastic){ choice = sample(1:nn,1)  } #stochastic (gradient) descent 
  if(!missing(Polyak.Ruppert.burn)){min.iter = max(min.iter,Polyak.Ruppert.burn)} #Polyak Ruppert
  
  
  # Step zero (latter half of loop that is needed now.)
  output$FunctionValue[1] = convex.function(beta[1,], data[choice,,drop=F], previous = NA, function.smoothing.factor = function.smoothing.factor)
  invH = diag(pp)
  gradient[1,] = gradient.function(beta = beta[1,], data = data[choice,,drop=F])
  
  # Loop
  for(i in 1:max.iter){
    # Stochastic piece: randomly choose single datapoint
    if(stochastic){
      choice = sample(1:nn,1)
      output$RandomChoice[i+1] = choice
    }

    # Get Newton Raphson search direction with approx inverse Hessian
    direction = - invH %*% gradient[i,]
    
    # Determine stepsize
    output$StepSize[i] = stepsize.function(iteration = i, 
                                             convex.function = convex.function, 
                                             direction.vector = direction, 
                                             current.gradient.vector = gradient[i,], 
                                             current.beta.vector = beta[i,], 
                                             data = data[choice,,drop=F],
                                             ...)
    
    # Update vector, i.e. Step down the hill	
    beta[i+1,] = beta[i,] + output$StepSize[i] * direction 
    
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
    
    # New gradient
    gradient[i+1,] = gradient.function(beta = beta[i+1,], data = data[choice,,drop=F])
    
    # BFGS update to approximate Inverse Hessian
    invH = BFGS.update(inverse.Hessian = invH,
                       old.beta = beta[i,],
                       new.beta = beta[i+1,],
                       old.gradient = gradient[i,],
                       new.gradient = gradient[i+1,]
    )
    
    # Stopping criteria
    if( (abs(output$FunctionValue[i] - output$FunctionValue[i+1])) < tol & (i >= min.iter)){
      return(list(betahat = output[i+1,(1:pp)+number.details], output=output[1:(i+1),]))
    }
  }
  print('Failed to converge')
  return(list(output=output))
}

BFGS.update = function(inverse.Hessian, old.beta, new.beta, old.gradient, new.gradient){
  d.gradient = new.gradient - old.gradient
  step = new.beta - old.beta
  denominator = as.numeric(crossprod(d.gradient, step))
  matrix1 = diag(length(new.beta)) - outer(d.gradient, step)/denominator
  updated.inverse.Hessian = matrix1 %*% inverse.Hessian %*% t(matrix1) + step %*% t(step) 
  # or tcrossprod(inverse.Hessian, matrix1) 
  return(updated.inverse.Hessian)
}

  