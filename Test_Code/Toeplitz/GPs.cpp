//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <cmath>
#include <array>
#include <chrono>
#include <memory>
#include <random>
#include <boost/range/irange.hpp>
#include <Eigen/Dense>
#include "GPs.h"
#include "utils.h"

//using namespace GP;
using Matrix = GP::Matrix;
using Vector = GP::Vector;


// Define kernel function for RBF
double GP::RBF::evalKernel(Matrix & x, Matrix & y, Vector & params, int n)
{
  switch (n)
    {
    case 0: return std::exp( -(x-y).squaredNorm() / (2.0*std::pow(params(0),2)));
    case 1: return (x-y).squaredNorm() / std::pow(params(0),3) * std::exp( -(x-y).squaredNorm() / (2.0*std::pow(params(0),2)));
    default: std::cout << "\n[*] UNDEFINED DERIVATIVE\n"; return 0.0;
    }
};

// Define distance kernel function for RBF
//
// REVISION:  Optimize w.r.t. theta = log(l) for stability.
//                 ==> .../ l^2  instead of .../ l^3
double GP::RBF::evalDistKernel(double d, Vector & params, int n)
{
  switch (n)
    {
    case 0: return std::exp( -d / (2.0*std::pow(params(0),2)));
    case 1: return d / std::pow(params(0),2) * std::exp( -d / (2.0*std::pow(params(0),2)));
    default: std::cout << "\n[*] UNDEFINED DERIVATIVE\n"; return 0.0;
    }
};


// Compute cross covariance between two input vectors using kernel parameters p
// (note: values of deriv > 0 correspond to derivative calculations)
void GP::RBF::computeCov(Matrix & K, Matrix & D, Vector & params, int deriv)
{
  // Get matrix input observation count
  auto n = static_cast<int>(D.rows());

  // Define lambda function to create unary operator (by clamping kernelParams argument)      
  auto lambda = [=,&params](double d)->double { return evalDistKernel(d, params, deriv); };

  for ( auto j : boost::irange(0,n) )
    {
      K.col(j) = (D.col(j)).unaryExpr(lambda);          
    }

  // Direct evaluation on individual coefficients
  //K = distMatrix.unaryExpr(lambda);
  
};



void GP::RBF::computeCrossCov(Matrix & K, Matrix & X1, Matrix & X2, Vector & params)
{
  // Get matrix input observation count
  //auto n = static_cast<int>(X1.rows());
  auto m = static_cast<int>(X2.rows());

   // Define lambda function to create unary operator (by clamping kernelParams argument)      
  auto lambda = [=,&params](double d)->double { return evalDistKernel(d, params, 0); };
  for ( auto j : boost::irange(0,m) )
    {
      K.col(j) = ((X1.rowwise() - X2.row(j)).rowwise().squaredNorm()).unaryExpr(lambda);          
    }
  
};


// Precompute distance matrix to avoid repeated calculations during optimization procedure
void GP::GaussianProcess::computeDistMat()
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  distMatrix.resize(n,n);  
  for ( int j=0 ; j < n; j++ )
    {
      distMatrix.col(j) = (obsX.rowwise() - obsX.row(j)).rowwise().squaredNorm();
    }
}


// Evaluate NLML for specified kernel hyperparameters p
// (note: alpha is used to share calculations with DNLML function)
double GP::GaussianProcess::evalNLML(const Vector & p)
{
  // Get matrix input observation count
  // Possibly not needed for NLML calculation ... ?
  auto n = static_cast<int>(obsX.rows());

  int paramCount = (*kernel).getParamCount();
  
  auto pcopy = static_cast<Vector>(p);
  if (!fixedNoise)
    {
      auto noise = static_cast<double>((pcopy.head(1))[0]);
      auto params = static_cast<Vector>( pcopy.tail(paramCount) );        

      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        params(i) = std::exp(params(i));

      // Compute covariance matrix and store Cholesky factor
      K.resize(n,n);
      (*kernel).computeCov(K, distMatrix, params, 0);
      cholesky = ( K + (noise+jitter)*Matrix::Identity(n,n) ).llt();
    }
  else
    {
      auto params = static_cast<Vector>( pcopy );

      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        params(i) = std::exp(params(i));

      // Compute covariance matrix and store Cholesky factor
      K.resize(n,n);
      (*kernel).computeCov(K, distMatrix, params, 0);
      cholesky = ( K + (noiseLevel+jitter)*Matrix::Identity(n,n) ).llt();
    }

  // Store alpha for DNLML calculation
  _alpha = cholesky.solve(obsY);

  // Compute NLML value
  auto NLML_value = 0.5*(obsY.transpose()*_alpha)(0)  +  0.5*static_cast<Matrix>(cholesky.matrixL()).diagonal().array().log().sum() + 0.5*std::log(2*PI);

  return NLML_value;
}


// Evaluate derivatives of NLML for specified kernel hyperparameters
// (note: alpha is used to retrieve calculations from NLML function)
void GP::GaussianProcess::evalDNLML(const Vector & p, Vector & g)
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  auto pcopy = static_cast<Vector>(p);

  int paramCount = (*kernel).getParamCount();
  
  Vector params;
  if (!fixedNoise)
    params = static_cast<Vector>( pcopy.tail(paramCount) );
  else
    params = static_cast<Vector>(pcopy);

  // ASSUME OPTIMIZATION OVER LOG VALUES
  for ( auto i : boost::irange(0,paramCount) )
    params(i) = std::exp(params(i));

  // Specify noise derivative if using trainable noise
  // and adjust array indices using shift accordingly
  int shift = 0;
  if (!fixedNoise)
    g[0] = -0.5 * ( _alpha*_alpha.transpose() - cholesky.solve(Matrix::Identity(n,n)) ).trace() ;
  else
    shift = 1;

  Matrix term = cholesky.solve(Matrix::Identity(n,n)) - _alpha*_alpha.transpose();
  Matrix dK_i(n,n);
  double trace;
  for (auto i : boost::irange(1,paramCount+1))
    {
      // Compute derivative of covariance matrix
      (*kernel).computeCov(dK_i, distMatrix, params, i);

      // Compute trace of full matrix
      //g[i-shift] = 0.5 * (term * dK_i ).trace() ;

      // Try computing only diagonal entries for trace calculation
      trace = 0.0;
      for (auto j : boost::irange(0,n))
        {
          trace += term.row(j)*dK_i.col(j);
        }
      
      g[i-shift] = 0.5*trace;
    }
}

// Define function for uniform sampling
Matrix GP::sampleUnif(double a, double b, int N)
{
  return (b-a)*(Eigen::MatrixXd::Random(N,1) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,1)) + a*Eigen::MatrixXd::Ones(N,1);
}


// Fit model hyperparameters
void GP::GaussianProcess::fitModel()
{
  // Get combined parameter/noise vector size
  int paramCount = (*kernel).getParamCount();
  int augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

  // Precompute distance matrix
  computeDistMat();
  
  // Specify initial parameter values for solver
  Vector x = static_cast<Vector>(Eigen::VectorXd::Ones(augParamCount));
  if (!fixedNoise)
    x[0] = 0.001;
  
  // Declare vector for storing gradient calculations
  Vector D(augParamCount);

  // Specify precision of minimization algorithm
  double SIG = 0.1;
  double EXT = 3.0;
  //int MAX = 20;
  int MAX = 30;
  //double INT = 0.01;
  double INT = 0.01;
  //int length = 1000;
  int length = 100;

  // Define restart count and hyperparameter bounds
  int restartCount = 10;
  double lb = -11.512925464970229;
  double ub = -11.512925464970229;
  double currentVal;
  double optVal = 1e9;
  Vector theta(augParamCount);
  Vector optParams(augParamCount);

  // Evaluate optimizer with various different initializations
  for ( auto i : boost::irange(0,restartCount) )
    {
      // Avoid compiler warning for unused variable
      (void)i;
      
      // Sample initial hyperparameter vector
      //theta = static_cast<Vector>(Eigen::VectorXd::Ones(augParamCount));
      theta = sampleUnif(lb, ub, augParamCount);
  
      // Optimize hyperparameters
      minimize::cg_minimize(theta, this , D, length, SIG, EXT, INT, MAX);

      currentVal = evalNLML(theta);

      if ( currentVal < optVal )
        {
          optVal = currentVal;
          optParams = theta;
        }
    }

  // Assign tuned parameters to model
  if (!fixedNoise)
    {
      noiseLevel = static_cast<double>((optParams.head(1))[0]);
      optParams = optParams.tail(augParamCount - 1);
      
      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        optParams(i) = std::exp(optParams(i));

      (*kernel).setParams(optParams);
    }
  else
    {
      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        optParams(i) = std::exp(optParams(i));

      (*kernel).setParams(optParams);
    }

  // Recompute covariance and Cholesky factor
  auto N = static_cast<int>(K.rows());
  Vector params = (*kernel).getParams();
  (*kernel).computeCov(K, distMatrix, params, 0);
  cholesky = ( K + (noiseLevel+jitter)*Matrix::Identity(N,N) ).llt();
  
};

// Compute predicted values
void GP::GaussianProcess::predict()
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  auto m = static_cast<int>(predX.rows());

  // Get optimized kernel hyperparameters
  Vector params = (*kernel).getParams();
  
  // Compute cross covariance for test points
  Matrix kstar;
  kstar.resize(n,m);
  (*kernel).computeCrossCov(kstar, obsX, predX, params);

  // Compute covariance matrix for test points
  Matrix kstarmat;
  kstarmat.resize(m,m);
  (*kernel).computeCrossCov(kstarmat, predX, predX, params);

  // Possible redundant calculations; should simplify...
  Matrix cholMat(cholesky.matrixL());
  Matrix alpha = cholesky.solve(obsY);
  Matrix v = kstar;
  cholMat.triangularView<Eigen::Lower>().solveInPlace(v);

  // Set predictive means/variances and compute negative log marginal likelihood
  predMean = kstar.transpose() * alpha;
  predCov = kstarmat - v.transpose() * v;

}


// Draw sample paths from posterior distribution
Matrix GP::GaussianProcess::getSamples(int count)
{
  // Get number of target points
  auto n = static_cast<int>(predX.rows());
  
  // Construct simple random generator engine from a time-based seed
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::normal_distribution<double> normal (0.0,1.0);

  // Assign i.i.d. random normal values to uVals
  Matrix uVals(n,count);
  for (auto i : boost::irange(0,n))
    {
      for (auto j : boost::irange(0,count))
          uVals(i,j) = normal(generator);
    }

  // Compute Cholesky factor L
  Matrix L = ( predCov + jitter*Matrix::Identity(static_cast<int>(predCov.cols()), static_cast<int>(predCov.cols())) ).llt().matrixL();
  
  // Draw samples using the formula:  y = m + L*u
  Matrix samples = predMean.replicate(1,count) + L*uVals;
  
  return samples;
}

