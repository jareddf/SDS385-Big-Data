# 
# Weighted Least Squares
#
wls.lu = function(y,X,W){
	# 
	# Solves for betahat vector in Weighted Least Squares using LU decomposition
	#

	# LU decomposition
	elu = expand(lu(t(X)%*%W%*%X))
	
	# q vector is one side of normal equation (see pdf)
 	q = t(X)%*%W%*%y

	#
	pp = dim(X)[2]
  	d = numeric(pp)
	betahat = numeric(pp)

	# Solve q = Ld via Gaussian elimination
  	for(j in 1:pp){
		index = (0:pp)[1:j]
		d[j] = (q[j] - sum(elu$L[j,index]*d[index]))/elu$L[j,j]
	}

	# Solve d = UB via Gaussian elimination
	for(j in pp:1){
		index = c(2:pp,0)[j:pp]
		betahat[j] = (d[j] - sum(elu$U[j,index]*betahat[index]))/elu$U[j,j]
	}

	#
	return(betahat)
}

wls.invert = function(y,X,W){
	#
	# Solves for betahat vector in Weighted Least Squares using matrix inversion
 	#
  solve(t(X)%*%W%*%X)%*%t(X)%*%W%*%y
}

wls.qr = function(y,X,W){
	#
	# Solves for betahat vector in Weighted Least Squares using QR decomposition
	# 

	pp = ncol(X)
	betahat = numeric(pp)

	# QR Decomposition
	W.sqrt = diag(sqrt(diag(W)))
	QR = qr(diag(sqrt(diag(W)))%*%X)

	# Solve RB = Q'W^.5 y via Gaussian elimination
	rhs = t(qr.Q(QR)) %*% W.sqrt %*% y
	R = qr.R(QR)
	for(j in pp:1){
		index = c(2:pp,0)[j:pp]
		betahat[j] = (rhs[j] - sum(R[j,index]*betahat[index]))/R[j,j]
	}
	return(betahat)
}


wls.sp = function(y,X,W){
  #
  # Solves for betahat vector in Weighted Least Squares using sparse matrix functions
  #
  X = Matrix(X, sparse=TRUE)
  solve(t(X)%*%W%*%X,t(X)%*%W%*%y)
}

#####
#####

binomial.log.likelihood = function(y,w,m=1,adjustment=.00001){
	#sum(dbinom(y,size=m,prob=w,log=TRUE))
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

steepest.descent = function(y,X,startbeta,m=1,max.iter=100,tol=.001,save.all=TRUE){
  ## Thanks to Jennifer for clarification in class! 
	# Assumes X's first column is already a column of 1's for intercept term. 
	nn = length(y)
	pp = ncol(X) 
	if(nrow(X) != nn){stop('Dimensionality of X,y not compatible')}

	# needed data structures. 
	output = data.frame(matrix(0,ncol=pp+3,nrow=max.iter+1))
	beta = matrix(0,ncol=pp,nrow=max.iter+1)
	names(output)[1:4] = c('Step','StepSize','LogLikelihood','Beta0')
	

	# Starting value. Defaults as 0's
	if(missing(startbeta)){startbeta=numeric(pp)}
	if(length(startbeta)!=pp){stop('startbeta not of length p')}
	beta[1,]=startbeta

	# Initial values
	output$LogLikelihood[1] = binomial.log.likelihood(y=y,w=predict.logistic(X,beta=beta[1,]),m=m)
	output$StepSize = NA
	output$StepSize[1] = .5
	output$Step = 0:max.iter
	

	# Loop
	for(i in 1:max.iter){
	  
	  # Find gradient at current beta
		w.hat = predict.logistic(X=X,beta=beta[i,])
		gradient = negative.logistic.gradient(y=y,X=X,w=w.hat,m=m)
    
		# Step down the hill	
		beta[i+1,] = beta[i,] - output$StepSize[i] * gradient 
		
		# Measure Log Likeihood of new beta
		output$LogLikelihood[i+1] = binomial.log.likelihood(y=y,w=predict.logistic(X,beta=beta[i+1,]),m=m)
		
		
		if(norm(beta[i,,drop=F]-beta[i+1,,drop=F]) < tol){
			output[,(1:pp)+3] <- beta
			return(list(betahat = beta[i+1,], output=output[1:(i+1),]))
		}
		output$StepSize[i+1] = .1/log(i+1) # *( 1- output$LogLikelihood[i]/output$LogLikelihood[i+1]) + .000001
	}
	output[,(1:pp)+3] <- beta
	print('Failed to converge')
	return(list(output=output))
}



gradient.descent = function(convex.function,gradient.function,starting.values,stepsize.function,max.iter=100,tol=.001){
	#
	# Optimizes over beta vector, input to convex.function
	#
	pp = length(starting.values) 

	# needed data structures. 
	output = data.frame(matrix(0,ncol=pp+3,nrow=max.iter+1))
	beta = matrix(0,ncol=pp,nrow=max.iter+1)
	names(output)[1:4] = c('Step','StepSize','FunctionValue','Beta0')
	

	# Starting value. Defaults as 0's
	beta[1,]=starting.values

	# Initial values
	output$FunctionValue[1] = convex.function(y=y,w=predict.logistic(X,beta=beta[1,]),m=m)
	output$StepSize = NA
	output$StepSize[1] = .5
	output$Step = 0:max.iter
	

	# Loop
	for(i in 1:max.iter){
	  
	  # Find gradient at current beta
		w.hat = predict.logistic(X=X,beta=beta[i,])
		gradient = negative.logistic.gradient(y=y,X=X,w=w.hat,m=m)
    
		# Step down the hill	
		beta[i+1,] = beta[i,] - output$StepSize[i] * gradient 
		
		# Measure Log Likeihood of new beta
		output$LogLikelihood[i+1] = binomial.log.likelihood(y=y,w=predict.logistic(X,beta=beta[i+1,]),m=m)
		
		
		if(norm(beta[i,,drop=F]-beta[i+1,,drop=F]) < tol){
			output[,(1:pp)+3] <- beta
			return(list(betahat = beta[i+1,], output=output[1:(i+1),]))
		}
		output$StepSize[i+1] = 1/log(i+1) * output$LogLikelihood[i]/output$LogLikelihood[i+1]
	}
	output[,(1:pp)+3] <- beta
	print('Failed to converge')
	return(list(output=output))
}

hessian_binomial.likelihood = function(X,w,m){
  #input w,m should be vectors! Or perhaps m can be a scalar
  if((length(w)!=length(m)) & length(m)>1){stop('Length of vector m is unacceptable')}
  t(X)%*%diag(m * w * (1-w))%*%X #inefficient???
}


Newtons.method = function(y,X,starting.vector,m=1,max.iter=100,tol=.001,StepSize=1){
  
  # needed data structures. 
  pp = length(starting.vector)
  output = data.frame(matrix(0,ncol=pp+3,nrow=max.iter+1))
  names(output)[1:4] = c('Step','StepSize','LogLikelihood','Beta0')
  beta = matrix(0,ncol=pp,nrow=max.iter+1)
  beta[1,]= starting.vector
  
  # Initial values
  output$LogLikelihood[1] = binomial.log.likelihood(y=y,w=predict.logistic(X,beta=beta[1,]),m=m)
  output$StepSize = StepSize
  output$Step = 0:max.iter
  
  # Loop
  for(i in 1:max.iter){
    
    # Find gradient and inverse Hessian at current beta
    w.hat = predict.logistic(X=X,beta=beta[i,])
    gradient = negative.logistic.gradient(y=y,X=X,w=w.hat,m=m)
    H = hessian_binomial.likelihood(X=X,w=w.hat,m=m)
    
    # Step down the hill, with curvature!	
    step = my.qr.solver(A = H, b = output$StepSize[i] * gradient )
    beta[i+1,]  =  beta[i,] - step
    
    # Measure Log Likeihood of new beta
    output$LogLikelihood[i+1] = binomial.log.likelihood(y=y,w=predict.logistic(X,beta=beta[i+1,]),m=m)
    
    
    if(norm(beta[i,,drop=F]-beta[i+1,,drop=F]) < tol){
      output[,(1:pp)+3] <- beta
      return(list(betahat = beta[i+1,], output=output[1:(i+1),]))
    }
  }
  output[,(1:pp)+3] <- beta
  print('Failed to converge')
  return(list(output=output))
}

# Convergence criteria: gradient ~ 0, beta_t ~ beta_{t-1}, L_t ~ L_{t-1}


my.qr.solver = function(A,b){
    #
    # Solves Ax = b, via QRx = b, Rx = Q'b
    # 
    pp = ncol(A)
    x = numeric(pp)
    
    # QR Decomposition
    QR = qr(A)
    R = qr.R(QR)
    rhs = t(qr.Q(QR))%*%b
    
    # Gaussian elimination
    for(j in pp:1){
      index = c(2:pp,0)[j:pp]
      x[j] = (rhs[j] - sum(R[j,index]*x[index]))/R[j,j]
    }
    return(x)
}


