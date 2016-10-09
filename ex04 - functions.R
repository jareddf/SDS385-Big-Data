# This file contains the functions needed to complete exercises 04 for SDS 385 - Big Data. 

###
### 1a --------------------------------------------------------------
###

gradient.descent = function(convex.function, gradient.function, stepsize.function, data, starting.values, function.smoothing.factor=1, max.iter=1000, min.iter=1, tol=.001, stochastic = F, Polyak.Ruppert.burn, minibatch.size, minibatch.frequency=50,...){
  #
  # Optimizes over beta vector, input to convex.function
  # Polyak Ruppert is ...?
  # Function smoothing factor is ...???
  
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
    if(missing(minibatch.size)){
      output$StepSize[i+1] = stepsize.function(iteration = i, 
                                               convex.function = convex.function, 
                                               direction.vector = -gradient, 
                                               current.gradient.vector = gradient, 
                                               current.beta.vector = beta[i,], 
                                               data = data[choice,,drop=F],
                                               ...)
    }
    else{
      if((i %% minibatch.frequency) == 0 | i == 1){
        minibatch = sample(1:nn, size = minibatch.size)
        avg.gradient = gradient.function(beta = beta[i,], data = data[minibatch,,drop=F])/minibatch.size #????
        output$StepSize[(i+1):max.iter] = stepsize.function(iteration = i, 
                                                 convex.function = convex.function, 
                                                 direction.vector = -avg.gradient, 
                                                 current.gradient.vector = avg.gradient, 
                                                 current.beta.vector = beta[i,], 
                                                 data = data[choice,,drop=F],
                                                 ...)
      }
    }

    
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
### 1b ---------------------------------------------------------------
###

adagrad = function(convex.function, gradient.function, stepsize.function, data, starting.values, function.smoothing.factor=1, max.iter=1000, min.iter=1, tol=.001, stochastic = T, Polyak.Ruppert.burn, ...){
  #
  # Optimizes over beta vector, input to convex.function
  # Polyak Ruppert smoothes betas
  # Function smoothing factor smoothes the convex function (likelihood)
  
  # Dimensions
  pp = length(starting.values) 
  nn = nrow(data)
  
  # Needed data structures. Returns "output" dataframe that includes each beta value, step size, etc. 
  number.details = 4
  beta = matrix(0, ncol = pp, nrow = max.iter + 1)
  output = data.frame(matrix(0, ncol = pp + number.details, nrow = max.iter + 1))
  names(output)[1:5] = c('Step','StepSize','RandomChoice','FunctionValue','Beta0')
  
  # Options
  if(!missing(Polyak.Ruppert.burn)){min.iter = max(min.iter,Polyak.Ruppert.burn)}
  
  # Initial values
  output$StepSize = NA
  output$RandomChoice = NA
  output$Step = 0:max.iter
  choice = 1:nn
  if(stochastic){ choice = sample(1:nn,1)  }
  G = numeric(pp)
  
  # Starting value. Defaults as 0's
  beta[1,] <- output[1, (1:pp) + number.details] <- starting.values
  
  
  # Step 0
  output$FunctionValue[1] = convex.function(beta[1,], data[choice,,drop=F], previous = NA, function.smoothing.factor = function.smoothing.factor)
  
  
  # Loop
  for(i in 1:max.iter){
    # Stochastic piece: randomly choose single datapoint
    if(stochastic){
      choice = sample(1:nn,1)
      output$RandomChoice[i+1] = choice
    }
    
    # Find gradient at current beta
    gradient = gradient.function(beta = beta[i,], data = data[choice,,drop=F])
    
    # Direction
    G = G + (gradient)^2
    direction = -gradient/(sqrt(G) + 1e-8)
    
    # Determine stepsize
    output$StepSize[i+1] = stepsize.function(iteration = i, 
                                               convex.function = convex.function, 
                                               direction.vector = direction, 
                                               current.gradient.vector = gradient, 
                                               current.beta.vector = beta[i,], 
                                               data = data[choice,,drop=F],
                                               ...)
 
    
    # Step down the hill	
    beta[i+1,] = beta[i,] + output$StepSize[i+1] * direction 
    
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


###
### 2 ----------------------------------------------------------------------------
### 


fast.sgd = function(
  convex.function
  , gradient.function
  , stepsize.function
  , data
  , starting.values
  , function.smoothing.factor=1
  , max.iter=1000
  , min.iter=1
  , tol=.001
 # , lik.penalty = 1
  #  , Polyak.Ruppert.burn
  , ...){
  
  #
  # adagrad
  # Optimizes over beta vector, input to convex.function
  # Polyak Ruppert smoothes betas
  # Function smoothing factor smoothes the convex function (likelihood)
  # Stochastic, but reads through dataset 1 line at a time, in order. 
  
  # Dimensions
  pp = length(starting.values) 
  nn = nrow(data)
  
  # Vectors for beta's
  beta <- oldbeta <- numeric(pp)
  
  # Diagonal of adagrad's approx Hessian
  adagrad.diag = numeric(pp)
  
  # Details dataframe that includes each value, step size, etc. 
  details =  c('Step','StepSize','RandomChoice','FunctionValue')
  output = data.frame(matrix(0, ncol = length(details), nrow = max.iter + 1))
  names(output) = details
  
  # Options
  #if(!missing(Polyak.Ruppert.burn)){min.iter = max(min.iter,Polyak.Ruppert.burn)}
  
  # Initial values
  output$StepSize = NA
  output$RandomChoice = NA
  output$Step = 0:max.iter
  
  ### Step 0  
  # Starting value. Defaults as 0's
  beta <- oldbeta <- starting.values
  # Function at starting values
  output$FunctionValue[1] = convex.function(beta, data[1,,drop=F], previous = NA, function.smoothing.factor = function.smoothing.factor)
              #lik.penalty = lik.penalty,
  
  # Loop
  for(i in 1:max.iter){
    
    # Stochastic piece: reads through in order ###! can remove this 
    output$RandomChoice[i+1] <- choice <- sample(1:nn,1)#i %% nn
    
    # Find gradient at current beta
    gradient = gradient.function(beta = beta, data = data[choice,,drop=F])#, lik.penalty = lik.penalty)
    
    # Direction
    adagrad.diag = adagrad.diag + (gradient)^2
    direction = -gradient/(sqrt(adagrad.diag) + 1e-8)
    
    # Determine stepsize
    output$StepSize[i+1] = stepsize.function() ###!!! constant
    # output$StepSize[i+1] = stepsize.function(iteration = i, 
    #                                          convex.function = convex.function, 
    #                                          direction.vector = direction, 
    #                                          current.gradient.vector = gradient, 
    #                                          current.beta.vector = beta[i,], 
    #                                          data = data[choice,,drop=F],
    #                                          ...)
    
    
    # Step down the hill
    oldbeta <- beta
    beta = beta + output$StepSize[i+1] * direction 
    
    # # Polyak-Ruppert averaging of betas. 
    # if(missing(Polyak.Ruppert.burn)){
    #   output[i+1,(1:pp)+number.details] <- beta[i+1,]
    # } 
    # else {
    #   output[i+1,(1:pp)+number.details] <- (1/(i+1)) * beta[i+1,]  +  (i/(i+1)) * output[i,(1:pp)+number.details]
    # }
    
    # Measure target function (Log Likeihood) of new beta - using only given point (and previous, for ). Requires averaged target function
    output$FunctionValue[i+1] = convex.function(
      beta = beta
      ,data = data[choice,,drop=F]
      ,previous = output$FunctionValue[i]
      ,function.smoothing.factor = function.smoothing.factor
      #,lik.penalty = lik.penalty
    )
    
    # Stopping criteria
    if( (abs(output$FunctionValue[i] - output$FunctionValue[i+1])) < tol & (i >= min.iter)){
      return(list(betahat = beta, output=output[1:(i+1),]))
    }
  }
  print('Failed to converge')
  return(list(output=output))
}




wrapper.fast.logreg.negative.gradient = function(beta, data, lik.penalty=1){
  # function for use within optimization (minimization) function. 
  # Input: beta vector, data matrix where first column is response variable Y
  # Output: negative gradient of log binomial likelihood for logistic regression
  
  # Strip Y vector off input data matrix
  X = data[,2:ncol(data),drop=F]
  y = data[,1,drop=F]
  
  # Get predict "success" probabilities
  w.hat = predict.logistic(X=X, beta=beta)
  
  # Compute and return negative gradient
  gradient = negative.penalized.logistic.gradient(y=y, X=X, w=w.hat, beta = beta, lik.penalty = lik.penalty)
  return(gradient)
}


wrapper.fast.logreg.negative.likelihood.EMA = function(beta, data, lik.penalty=1, previous = NA, function.smoothing.factor = 1){
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
  lik = -penalized.binomial.log.likelihood(y=Y, w=w.hat, beta=beta, lik.penalty=lik.penalty)
  
  # Exponential moving average, alpha = function.smoothing.factor. But l1 = l1
  if(is.na(previous)){
    return(lik)
  }
  else{
    return( function.smoothing.factor * lik + (1 - function.smoothing.factor) * previous )
  }
}

library(Rcpp)
cppFunction('
            double logBinomialLikelihood(NumericVector y, NumericVector w, NumericVector beta, double penalty = 0, double adjustment = .00001) {
            
            int nn = y.size();
            int pp = beta.size();
            double sumout = 0;
            double sumbeta2 = 0;
            NumericVector m(nn);
            
            for(int i = 0; i < nn; i++) {
            m[i] = 1;
            sumout += y[i]*log(w[i] + adjustment) + (m[i] - y[i])*log(1 - w[i] + adjustment);
            }
            
            for(int j = 0; j < pp; j++){
            sumbeta2 +=  beta[j] * beta[j];
            }
            
            sumout = sumout - penalty * sumbeta2;
            return sumout;
            }
            ')



negative.penalized.logistic.gradient = function(y, X, w, beta, lik.penalty=1){
  -( t(X)%*%(y - w) ) + 2*lik.penalty*sum(beta) 
}


