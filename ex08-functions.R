# Functions for ex08
soft.threshold = function(y,lambda){
  yyy = abs(y) - lambda
  out = ifelse(y<0,-1,1)*ifelse(yyy>0,yyy,0)
  return(out)
}

makeD2_sparse = function (dim1, dim2)  {
  require(Matrix)
  D1 = bandSparse(dim1 * dim2, m = dim1 * dim2, k = c(0, 1), diagonals = list(rep(-1, dim1 * dim2), rep(1, dim1 * dim2 - 1)))
  D1 = D1[(seq(1, dim1 * dim2)%%dim1) != 0, ]
  D2 = bandSparse(dim1 * dim2 - dim1, m = dim1 * dim2, k = c(0, dim1), diagonals = list(rep(-1, dim1 * dim2), rep(1,dim1 * dim2 - 1)))
  return(rBind(D1, D2))
}

linearsolve.jacobi = function(A,b,max.iter=100,tol=.01){
 #Separate diagonal and "other"
 D = diag(A)
 R = A
 diag(R) = 0

 # Precache
 D.inv = 1/D
 
 # Initial value
 x.old <- numeric(ncol(A))
 
 for(i in 1:max.iter){
   # Jacobi update
   x = D.inv * (b - R %*% x.old) #used to have this as matrix mulitplication... thanks Giorgio!
   
   # Check convergence
   error = max(abs(x - x.old))
   if(error < tol){
     return(x)
   }
   
   # If not converged, 
   x.old = x
 }
 print('Failed to Converge')
 return(x)
}


ADMM_graph.fused.lasso = function(y,D,lambda,eta,max.iter=1000,tol = .01){
  # minimizes .5 * ||y-x||_2^2 + lambda*||DX||_1, but observations are weighted: sum(eta*(y-x)^2)
  # From Section 5 of Tansey, Koyejo, Poldrack and Scott. 
  # See https://arxiv.org/pdf/1411.6144v1.pdf
  
  # INFO:
  # eta are weights for each datapoint.
  # The stepsize is "a" in the paper. 
  # D is oriented edge matrix. 
  
  # eta Weights preprocessing
  if(missing(eta)){eta = rep(1,length(y))}
  if(length(eta)!=length(y)){stop('Length of chosen eta weights is wrong')}
  
  # Initialize values/vectors
  x = y
  a = 2*lambda
  u <- z <- z_old <- x*0
  r <- s <- s_old <- t <- numeric(nrow(D))
  
  # Precache Large matrix
  tD = t(D)
  IpDTD = tD %*% D
  diag(IpDTD) = diag(IpDTD) + 1
  #IpDTD.inv = solve(IpDTD,sparse=T)
  
  for(k in 1:max.iter){
    
    # Update x
    x = (eta*y + a*(z-u))/(eta + a)
    
    # Update r, soft-thresholding function
    # r = soft.threshold(s-t,lambda/a) not working...?
    temp = abs(s-t) - lambda/a
    r = sign(s-t)*ifelse(temp>0,as.vector(temp),0)

    # Update z
    #z = IpDTD.inv %*% (x + u + tD%*%(r+t))  # alternative
    z = solve(IpDTD, x + u + tD%*%(r+t), sparse = T)
    
    #Update s 
    s = D %*% z
    
    # Update dual variables u,t
    u = u + x - z
    t = t + r -s
    
    # Check for convergence
    norm_primal.residual = sqrt( sum((x - z)^2) +  sum((r - s)^2) ) 
    norm_dual.residual = sqrt( sum((a * (z - z_old))^2) + sum((a * (s - s_old))^2) )
    if(norm_dual.residual < tol & norm_dual.residual < tol){
      print(paste(k,'iterations'))
      return(x)
    }
    
    # correct stepsize
    if(norm_primal.residual >= 5*norm_dual.residual){
      a = a * 2
      u = u / 2
      t = t / 2
    }
    if(norm_dual.residual >= 5*norm_primal.residual){
      a = a / 2
      u = u * 2
      t = t * 2
    }
    
    # Roll over old values
    s_old = s
    z_old = z
  }
  
  # if not converged:
  print('Failed to Converge')
  return(x)
}


