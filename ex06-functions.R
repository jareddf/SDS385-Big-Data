# Functions need for ex06-solutions.R

#Soft Thresholding function from ex05
soft.threshold = function(y,lambda){
  yyy = abs(y) - lambda
  out = ifelse(y<0,-1,1)*ifelse(yyy>0,yyy,0)
  return(out)
}

# Robbins Monro step size, from ex02
step.RobbinsMonro = function(t,t0=1,C,alpha){
  if(alpha > 1 | alpha < .5){stop('check step size "a" is in reasonable range')}
  if(C <= 0){stop('check step size "C" - currently not positive')}
  C*(t+t0)^(-alpha)
}

# Lasso - Proximal Gradient Method
lasso.proximalgradient = function(y, X, lambda, stepsize.function, max.iter=1000, tol = 1e-5, min.iter=5, beta0, ...){
  # Solves for coefficients of lasso regression
  # X,y should be standardized (centered and scaled). This removes need for intercept's column of ones
  # lambda is lasso penalty
  
  # Dimensions
  nn = length(y)
  pp = ncol(X)
  if(nn != nrow(X)){stop('Check dimensions of X,y')}
  if(sum(X[,1] != 1) == 0){stop('First column of ones is not needed in X')}
  
  # Useful Data structures 
  objective = numeric(max.iter + 1) #store value of objective function at each step
  tX = t(X)
  
  
  # Initialize at jittered LS solution if no starting point given 
  if(missing(beta0)){
    beta = jitter(solve(tX%*%X,tX%*%y)) #* 0
  } else {
    beta = beta0
  }
  
  # t = 0 case, initialize at LS estimate
  if(length(beta)!=pp){stop('check inital beta values.')}
  residuals = y - X%*%beta
  objective[1] = sum(residuals^2) + lambda*sum(abs(beta))
  
  for(t in 1:max.iter){
    
    # Calculate step size gamma (can be constant)
    gamma = stepsize.function(t,...)
    
    # Proximal method
    u = beta +  gamma * tX %*% residuals # took off constant of 2
    beta = soft.threshold(u,lambda*gamma)
    
    #
    residuals = y - X%*%beta
    objective[t+1] = sum(residuals^2) + lambda*sum(abs(beta))
    if(abs(objective[t+1] - objective[t]) < tol & t > min.iter){
      return(list(beta=beta,objective=objective[1:(t+1)]))
    }
  }
  
  ## If not converged
  print('Failed to Converge')
  return(list(beta=beta,objective=objective))
}







# ------------------------------------------------------------------------------------------







# Lasso - Accelerated Proximal Gradient Method
lasso.accelerated.proximalgradient = function(y, X, lambda, stepsize.function, max.iter=1000, tol = 1e-5, min.iter=5, beta0, ...){
  # Solves for coefficients of lasso regression
  # X,y should be standardized (centered and scaled). This removes need for intercept's column of ones
  # lambda is lasso penalty
  
  # Dimensions
  nn = length(y)
  pp = ncol(X)
  if(nn != nrow(X)){stop('Check dimensions of X,y')}
  if(sum(X[,1] != 1) == 0){stop('First column of ones is not needed in X')}
  
  # Data structures
  objective = numeric(max.iter + 1) #to save each step's progress
  tX = t(X)
  
  # Initialize at jittered LS solution if no starting point given 
  if(missing(beta0)){
    beta = jitter(solve(tX%*%X,tX%*%y)) #* 0
  } else {
    beta = beta0
  }
  
  # t = 0 case, initialize at LS estimate
  if(length(beta)!=pp){stop('check inital beta values.')}
  residuals = y - X%*%beta
  objective[1] = sum(residuals^2) + lambda*sum(abs(beta))
  
  # accelerated needs
  extrapolation <- beta.old <- beta.new <- beta 
  s.new <- s.old <- 1
  
  for(t in 1:max.iter){
    
    # Calculate step size gamma (can be constant)
    gamma = .0001#stepsize.function(t,...)
    
    # Accelerated Proximal method
    u = extrapolation + 2 * gamma * tX %*% (y - X%*%extrapolation)
    beta.new = soft.threshold(u,lambda*gamma)
    s.new = .5*(1 + sqrt(1 + 4*s.old^2))
    extrapolation = beta.old + ((s.old - 1)/(s.new)) * (beta.new - beta.old)
    
    # increment s, beta
    beta.old = beta.new
    s.old = s.new
    
    # Record current value of objective function
    residuals = y - X%*%beta.new
    objective[t+1] = sum(residuals^2) + lambda*sum(abs(beta.new))
    
    # end loop if tolerance met
    if(abs(objective[t+1] - objective[t]) < tol & t > min.iter){
      return(list(beta=beta.new,objective=objective[1:(t+1)]))
    }
  }
  
  ## If not converged
  print('Failed to Converge')
  return(list(beta=beta,objective=objective))
}


