//Thanks to James, Kevin, and Jennifer's code for guidance. 

#include <RcppEigen.h>
#include <algorithm>    // std::max

using namespace Rcpp;
//using Eigen::Map;
using Eigen::MatrixXd;
//using Eigen::LLT;
//using Eigen::Lower;
//using Eigen::MatrixXi;
//using Eigen::Upper;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::SparseVector;
typedef Eigen::MappedSparseMatrix<double>  MapMatd;
typedef SparseVector<double>::InnerIterator IIvector;
//typedef Map<MatrixXi>  MapMati;
//typedef Map<VectorXd>  MapVecd;
//typedef Map<VectorXi>  MapVeci;

// [[Rcpp::depends(RcppEigen)]] 
// [[Rcpp::export]]
SEXP sgd_logit_sparse(MapMatd X, VectorXd y, VectorXd m, VectorXd starting_values, 
                      double stepsize = 1.0, 
                      double likelihood_penalty = 0.0,
                      double smooth = 0.001,
                      int npass = 1
){
  // Description here (to be continued...)
  // X is the design matrix in column-major format
  // Returns EMA of likelihood (exponentially moving average, which uses "smooth").
  // smooth is likelihood smoothing factor
  
  // Dimensions
  int nn = X.cols(); //number of obs
  int pp = X.rows(); //number of features
  
  // Setup Data structures
  SparseVector<double> xi(pp);
  VectorXd beta(pp), adagrad_vector(pp);
  VectorXi previous_update(pp);
  VectorXd negative_likelihood(npass*nn);
  double XBeta, eXBeta, yhat, gradient, error, adjustment;
  double intercept = 0.0;
  double adagrad_intercept = 0.0;
  double nll_avg = 0.0;
  int j;
  int iteration = 0; 
  
  // Initial values
  for(int j=0; j<pp; j++){
    beta[j] = starting_values[j];
    adagrad_vector[j] = 1e-5;
    previous_update[j] = 0;
  }
  
  //LOOP - starts at 1, so that likelihood of starting values is iteration 0
  for(int pass=0; pass<npass; pass++){
    for(int i=0; i<nn; i++){
      
      // extract sparse vector
      xi = X.innerVector(i);
      
      //obtain needed pieces for likelihood gradient 
      XBeta = intercept + xi.dot(beta);
      eXBeta = exp(XBeta);
      yhat = m[i] * eXBeta / (1 + eXBeta);
      
      //Likelihood calculation - Thanks to James for the idea to get around iteration 0 issue
      nll_avg =  -(y[i] * XBeta - m[i] * log(1 + eXBeta))*smooth + (1-smooth)*nll_avg;
      negative_likelihood[iteration] = nll_avg; 
      
      // Update intercept //not penalized!
      error = y[i] - yhat; // gradient at intecept is -error
      adagrad_intercept += error * error; 
      intercept += stepsize * error / sqrt(adagrad_intercept); //
        
        //Lazy updating
      for (IIvector it(xi); it; ++it) {
        
        // identify nonzero element
        j = it.index();
        
        //// accumlated penalty from skipped updates
        double skips = iteration - previous_update[j] - 1;
        //if(skips > 5){skips = 5;} // Jennifer's suggestion 
        
        // accumulated penalty         
        // Version in my pdf. But this produce NaN's frequently... 
        // beta(t+k) = (1 - stepsize  * 2 * penalty / sqrt(G))^skips beta[j]
    //    adjustment = 1 - 2 * stepsize * likelihood_penalty / sqrt(adagrad_vector[j]);
    //    if(adjustment < 0){adjustment = 0;} // second hack to protect us from the first. Thanks James!
    //    beta[j] = pow(adjustment,skips) * beta[j];
        
        // Hybrid version between mine and James'
        adjustment = 1 - 2 * stepsize * likelihood_penalty * skips / sqrt(adagrad_vector[j]);
        if(adjustment < 0){adjustment = 0;} // second hack to protect us from the first. Thanks James!
        beta(j) = beta(j) * adjustment;
        
        //// Actual update:
        // penalized gradient = -(y - yhat) * x + 2 * penalty * beta
        gradient = - error * it.value() + 2 * likelihood_penalty * beta[j];
        
        // Update adagrad_vector // I think this should include the gradient of the penalty
        adagrad_vector[j] += gradient * gradient; 
        
        //Update Beta - walk down the hill //adagrad_vector //constant stepsize
        beta[j] -= stepsize*gradient/sqrt(adagrad_vector[j]);
        
        // record update was given
        previous_update[j] = iteration;
        
      } //close lazy updating loop (x, it)
      
      //increment iteration
      ++iteration;
      
    } //close observation-level loop (i)
  } //close number of passes loop (pass)
  
  for(int j=0; j<nn; j++){
    //cap lazy updates. 
  }
  // Conclude
  return List::create(Named("intercept") = intercept,
                      Named("beta") = beta,
                      Named("nll") = negative_likelihood);
}

