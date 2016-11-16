soft.threshold = function(y,lambda){
  yyy = abs(y) - lambda
  out = ifelse(y<0,-1,1)*ifelse(yyy>0,as.vector(yyy),0)
  return(out)
}

L1norm = function(vector){
  sum(abs(vector))
}

L2norm = function(vector){
  sqrt(sum((vector^2)))
}

custom.binary.search = function(input.vector,constraint,max.iter=100,tol=.01){
  minimum = 0
  maximum = max(abs(input.vector))
  
  ### First Try
  # Obtain L1 norm of initial estimate
  S = soft.threshold(input.vector,0)
  output.vector = S/L2norm(S)
  l1norm = L1norm(output.vector)
  
  # Check if reasonable
  if(l1norm <= constraint){
    return(output.vector)
  } 
  
  # while Loop? for with max.iter?
  for(iter in 1:max.iter){
    delta = mean(c(minimum, maximum))
    S = soft.threshold(input.vector,delta)
    output.vector = S/L2norm(S)
    l1norm = L1norm(output.vector)
    
    if(l1norm > constraint){
      minimum = delta
    }
    
    if(l1norm <= constraint){
      maximum = delta
      if((constraint - l1norm) < tol){
        return(output.vector)
      }
    } 
  }

  stop('Failed to converge in binary search')
}

PMD.singlefactor = function(X,cv,cu,tol=1,max.iter = 1000){
  # Description: Single factor penalized matrix decomposition (Witten, Tibsharani, Hastie, 2009), Algorithm 3
  
  #Checks
  if(!is.matrix(X)){stop('X is not a matrix')}
  
  # Dimensions
  nn = nrow(X)
  pp = ncol(X)
  
  ### STEP 1
  # Initialize v to have L2 norm = 1
  v_old = rep(1/pp,pp) # chose normalized 1 vector.... ????
  # I sneak u in there too
  u_old = rep(0,nn)
  
  ### STEP 2 - iterate until convergence
  for(iter in 1:max.iter){
    # Step 2a
    u = custom.binary.search(input.vector = X%*%v_old,constraint = cu)
    
    # Step 2b
    v = custom.binary.search(input.vector = t(X)%*%u,constraint = cv)
    
    # Check convergence - ADHOC!!!!@&T&^!$&#^
    if( L1norm(c(u - u_old,v - v_old)) < tol){
      d = t(u) %*% X %*% v
      return(list(d=d,u=u,v=v))
    }
    
    # Cycle variables
    u_old = u
    v_old = v
  }
  
  # Reaching this point means convergence failed. 
  print('Failed to Converge')
  return(NULL)
}

PMD.multifactor = function(k,X,cv,cu,...){
  # PMD for multiple factors
  
  # Caution
  if((k %% 1)!=0){stop('k must be an integer')}
  if(k > min(dim(X))){stop('k too large')}
  
  # Output object
  d = numeric(k)
  U = matrix(0,ncol=k,nrow=nrow(X))
  V = matrix(0,ncol=k,nrow=ncol(X))
  row.names(V) = colnames(X)
  
  for(i in 1:k){
    # Single factor on current residuals
    duv = PMD.singlefactor(X,cv,cu,...)
    
    # create new
    X = X - as.numeric(duv$d) * duv$u %*% t(duv$v)
    
    # add to Matrices
    d[i] = duv$d
    U[,i] = duv$u
    V[,i] = duv$v
  }
  
  return(list(d=d,U=U,V=V))
}
  
  
  
  
  
  