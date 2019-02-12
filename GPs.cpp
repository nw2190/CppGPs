//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <cmath>
#include <array>
#include <vector>
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
std::vector<Matrix> GP::RBF::computeCov(Matrix & K, Matrix & D, Vector & params, bool evalGrad)
{
  // Define lambda function to create unary operator (by clamping kernelParams argument)      
  auto lambda = [=,&params](double d)->double { return evalDistKernel(d, params, 0); };
  K.noalias() = D.unaryExpr(lambda);

  // Compute gradient list if "evalGrad=true"
  std::vector<Matrix> gradList;
  if ( evalGrad )
    {
      Matrix dK_i = ( D.array() * (1/std::pow(params(0),2)) * K.array() ).matrix();
      gradList.push_back(dK_i);
    }

  return gradList;
  
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

// Evaluate NLML [public interface]
double GP::GaussianProcess::computeNLML(const Vector & p, double noise)
{
  // Get hyperparameter count ( + noise parameter )
  int paramCount = (*kernel).getParamCount();
  int augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

  // Compute log-hyperparameters
  Vector logparams(augParamCount);
  //logparams(0) = std::log(noiseLevel);
  if ( !fixedNoise )
    {
      logparams(0) = std::log(noise);
      for ( auto i : boost::irange(1,augParamCount) )
        logparams(i) = std::log(p(i-1));
    }
  else
    {
      for ( auto i : boost::irange(0,paramCount) )
        logparams(i) = std::log(p(i));
    }

  // Evaluate NLML using log-hyperparameters
  return evalNLML(logparams);
}

// Evaluate NLML with default noise level [public interface]
double GP::GaussianProcess::computeNLML(const Vector & p)
{
  return computeNLML(p, noiseLevel);
}

// Evaluate NLML with default noise level [public interface]
double GP::GaussianProcess::computeNLML()
{
  auto params = (*kernel).getParams();
  return computeNLML(params, noiseLevel);
}


// Evaluate NLML for specified kernel hyperparameters p
double GP::GaussianProcess::evalNLML(const Vector & p, Vector & g, bool evalGrad)
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());

  // Get hyperparameter count ( + noise parameter )
  int paramCount = (*kernel).getParamCount();
  int augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

  // ASSUME OPTIMIZATION OVER LOG VALUES
  auto pcopy = static_cast<Vector>(p);
  for ( auto i : boost::irange(0,augParamCount) )
    pcopy(i) = std::exp(pcopy(i));

  double noise;
  Vector params;
  if (!fixedNoise)
    {
      noise = static_cast<double>((pcopy.head(1))[0]);
      params = static_cast<Vector>( pcopy.tail(paramCount) );        
    }
  else
    {
      noise = noiseLevel;
      params = static_cast<Vector>( pcopy );
    }

  // Compute covariance matrix and store Cholesky factor
  K.resize(n,n);
  auto gradList = (*kernel).computeCov(K, distMatrix, params, evalGrad);
  cholesky = ( K + (noise+jitter)*Matrix::Identity(n,n) ).llt();

  // Store alpha for DNLML calculation
  _alpha = cholesky.solve(obsY);

  // Compute NLML value
  auto NLML_value = 0.5*(obsY.transpose()*_alpha)(0) + static_cast<Matrix>(cholesky.matrixL()).diagonal().array().log().sum() + 0.5*n*std::log(2*PI);

  if ( evalGrad )
    {

      // Precompute the multiplicative term in derivative expressions
      Matrix term = cholesky.solve(Matrix::Identity(n,n)) - _alpha*_alpha.transpose();
      
      // Compute gradient for noise term if 'fixedNoise=false'
      int index = 0;
      if (!fixedNoise)
        {
          // Specify gradient of white noise kernel
          Matrix dK_noise = noise*Matrix::Identity(n,n);
          
          // Compute trace of full matrix
          g(index) = 0.5 * (term * dK_noise ).trace() ;
          index++;
        }

      // Compute gradients with respect to kernel hyperparameters
      for (auto dK_i = gradList.begin(); dK_i != gradList.end(); ++dK_i) 
        {
          // Compute trace of full matrix
          g(index) = 0.5 * (term * (*dK_i) ).trace() ;
          index++;
          /*
          // Try computing only diagonal entries for trace calculation
          trace = 0.0;
          for (auto j : boost::irange(0,n))
            {
              trace += term.row(j)*((*dK_i).col(j));
            }

          g(index) = 0.5*trace;
          index++;
          */
        }
    }

  return NLML_value;
  
}


// Define simplified interface for evaluating NLML without gradient calculation
double GP::GaussianProcess::evalNLML(const Vector & p)
{
  Vector nullGrad(0);
  return evalNLML(p,nullGrad,false);
}


// Define function for uniform sampling
Matrix GP::sampleUnif(double a, double b, int N)
{
  return (b-a)*(Eigen::MatrixXd::Random(N,1) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,1)) + a*Eigen::MatrixXd::Ones(N,1);
}

// Define function for uniform sampling [Vectors]
Vector GP::sampleUnifVector(Vector lbs, Vector ubs)
{
  auto n = static_cast<int>(lbs.rows());
  Vector sampleVector(n);
  Vector samples = Eigen::MatrixXd::Random(n,1);
  double a;
  double b;
  for ( auto i : boost::irange(0,n) )
    {
      // NOTE: This can easily be vectorized
      a = lbs(i);
      b = ubs(i);
      sampleVector(i) = (b-a)*(samples(i) * 0.5 + 0.5*samples(i)) + a*samples(i);
    }
  return sampleVector;
}


// Fit model hyperparameters
void GP::GaussianProcess::fitModel()
{
  // Get combined parameter/noise vector size
  int paramCount = (*kernel).getParamCount();
  int augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

  // Precompute distance matrix
  computeDistMat();

  // Declare vector for storing gradient calculations
  Vector g(augParamCount);

  // Specify precision of minimization algorithm
  int MAX = 10;
  int length = 100;
  double INT = 0.01;
  double SIG = 0.1;
  double EXT = 3.0;

  // Define restart count for optimizer
  int restartCount = 20;

  // Convert hyperparameter bounds to log-scale
  auto lbs = (lowerBounds.array().log()).matrix();
  auto ubs = (upperBounds.array().log()).matrix();

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
      theta = sampleUnifVector(lbs, ubs);

      // Optimize hyperparameters
      minimize::cg_minimize(theta, this, g, length, SIG, EXT, INT, MAX);

      currentVal = evalNLML(theta);

      if ( currentVal < optVal )
        {
          optVal = currentVal;
          optParams = theta;
        }
    }

  // ASSUME OPTIMIZATION OVER LOG VALUES
  for ( auto i : boost::irange(0,augParamCount) )
    optParams(i) = std::exp(optParams(i));
  
  // Assign tuned parameters to model
  if (!fixedNoise)
    {
      noiseLevel = static_cast<double>((optParams.head(1))[0]);
      optParams = static_cast<Vector>(optParams.tail(augParamCount - 1));
    }
  
  (*kernel).setParams(optParams);

  // Recompute covariance and Cholesky factor
  auto N = static_cast<int>(K.rows());
  Vector params = (*kernel).getParams();
  auto nullGradList = (*kernel).computeCov(K, distMatrix, params);
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

  //std::cout << params << std::endl;
  
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
  Matrix L = ( predCov + (noiseLevel+jitter)*Matrix::Identity(static_cast<int>(predCov.cols()), static_cast<int>(predCov.cols())) ).llt().matrixL();

  // Draw samples using the formula:  y = m + L*u
  Matrix samples = predMean.replicate(1,count) + L*uVals;
  
  return samples;
}

