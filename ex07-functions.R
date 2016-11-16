soft.threshold = function(y,lambda){
  yyy = abs(y) - lambda
  out = ifelse(y<0,-1,1)*ifelse(yyy>0,yyy,0)
  return(out)
}




admm_lasso = function(Y,X,lambda,rho,max.iter = 1000,tol=.001){
  # ADMM implementation of Lasso
  # minimizes || Xbeta - y  ||22 + lambda*||beta||1
  
  # Get dimensions
  pp = ncol(X)
  nn = nrow(X)
  if(nn != length(Y)){stop('Check X,y dimensions')}
  
  # Establish objects
  z <- u <- beta <- jitter(numeric(pp))
  objective = numeric(max.iter + 1)

  # Adjust tolerance, to avoid sqrt's later
  tol = tol^2
  
  # precache
  XtX = t(X) %*% X
  XtY = t(X) %*% Y
  ridge = solve(XtX + rho*diag(pp))

  # Step 0
  objective[1] = .5 * sum((X%*%beta - Y)^2) + lambda * sum(abs(beta))
  
  for(i in 1:max.iter){
    
    # Update beta
    beta = ridge %*% (XtY + rho*(z-u))
      
    # Update beta surrogate
    z_old = z
    z = soft.threshold(beta + u, lambda/rho)
    
    # Update error
    u = u + beta - z
    
    # evaluate objective 
    objective[i+1] = .5 * sum((X%*%beta - Y)^2) + lambda * sum(abs(beta))
    
    # Check tolerance
    r = beta - z
    primal.error = t(r)%*%r # removed sqrt - just adjust tolerance
    s = - rho * (z - z_old)
    dual.error = t(s)%*%s # removed sqrt
    if(primal.error < tol & dual.error < tol){
      return(list(beta = beta, objective = objective[1:(i+1)]))
    }
    
  }
  
  # If failed to converge by max.iter
  print('Failed to Converge')
  return(list(beta = beta, objective = objective))
}