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
//using Eigen::VectorXi;
using Eigen::SparseVector;
typedef Eigen::MappedSparseMatrix<double>  MapMatd;
typedef SparseVector<double>::InnerIterator IIvector;
//typedef Map<MatrixXi>  MapMati;
//typedef Map<VectorXd>  MapVecd;
//typedef Map<VectorXi>  MapVeci;

// [[Rcpp::depends(RcppEigen)]] 
// [[Rcpp::export]]
SEXP sgd_logit_dense(MatrixXd X, VectorXd y, VectorXd m, VectorXd starting_values, 
                    double stepsize = 1.0, 
                    int iterations = 1000, 
                    double likelihood_penalty = 0.0,
                    double likelihood_smoothing_factor = 0.001
){
  // Description here (to be continued...)
  // X is the design matrix in column-major format
  // Returns EMA of likelihood (exponentially moving average, which uses likelihood_smoothing_factor).
  
  // Dimensions
  int nn = X.cols(); //number of obs
  int pp = X.rows(); //number of features
  
  // Setup Data structures
  VectorXd beta(pp), adagrad(pp), gradient(pp), direction(pp);
  VectorXd negative_likelihood(iterations + 1);
  double XBeta; 
  
  // Initial values
  int choice = 0; 
  for(int j=0; j<pp; j++){
    beta[j] = starting_values[j];
    adagrad[j] = 1e-5;
  }
  
  // Initial Likelihood Calculation
  //x = X.col(1);
  XBeta =   (X.col(0)).dot(beta);
  negative_likelihood[0] = -(y[0] * XBeta - m[0] * log(1 + exp(XBeta)));
  
  //LOOP - starts at 1, so that likelihood of starting values is iteration 0
  for(int iter=1; iter<(iterations+1); iter++){
    //Stochastic choice of observation. In this case we just plow through. 
    choice = iter % nn; 
    
    //Gradient
    XBeta = (X.col(choice)).dot(beta);
    gradient = -X.col(choice) * (y[choice] - (1/(1+exp(-XBeta))));// + 2 * likelihood_penalty * beta.sum();
                                   
    //Direction
    adagrad += gradient.cwiseProduct(gradient); 
    for(int j=0; j<pp; j++){
      direction[j] = -gradient[j]/sqrt(adagrad[j]);
    }
    //direction = -gradient/(sqrt(adagrad));
    
    //Constant Stepsize 
    //stepsize = stuff!
    
    //Update Beta - walk down the hill
    beta += stepsize*direction;
    
    //Compute Likelihood for chosen datapoint with current beta. 
    XBeta =   (X.col(choice)).dot(beta);
    negative_likelihood[iter] = (1-likelihood_smoothing_factor) * negative_likelihood[iter-1] - likelihood_smoothing_factor*(y[choice] * XBeta - m[choice] * log(1 + exp(XBeta)));
  
  
  }
  
  // Conclude
  return List::create(Named("beta") = beta,
                      Named("nll") = negative_likelihood);
}





