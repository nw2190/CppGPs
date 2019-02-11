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

/*
int GP::RBF::getParamCount()
{
  return paramCount;
}

Vector GP::RBF::getParams()
{
  return kernelParams;
}

void GP::RBF::setParams(Vector params)
{
  kernelParams = params;
}
*/

/*
GP::Kernel::Kernel(Vector & p, int n)
{
  kernelParams = p;
  paramCount = n;
};
*/

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
//
// ==> .../ l^2  instead of .../ l^3
//
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
  //K = distMatrix.unaryExpr(lambda);
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
  //distMatrix.resize(n);  
  //distMatrix = (obsX.rowwise() - obsX.row(0)).rowwise().squaredNorm();
  
}


/*
// Compute cross covariance between two input vectors using kernel parameters p
// (note: values of deriv > 0 correspond to derivative calculations)
void GP::GaussianProcess::computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p, int deriv, bool useDistMat=false)
{
  // Get matrix input observation count
  auto n = static_cast<int>(v1.rows());
  auto m = static_cast<int>(v2.rows());

  // Evaluate using vectorized implementation of distance kernel,
  // or apply standard binary kernel function component-wise
  if (useDistKernel)
    {

      if (useDistMat)
        {
          //std::cout << "\nUSING DISTANCE MATRIX\n";
          // Define lambda function to create unary operator (by clamping kernelParams argument)      
          auto lambda = [=,&p](double d)->double { return (*distKernel)(d, p, deriv); };
          //int j;
          //#pragma omp parallel for
          for ( int j=0 ; j < m; j++ )
            {
              M.col(j) = (distMatrix.col(j)).unaryExpr(lambda);          
            }
          //M = distMatrix.unaryExpr(lambda);
        }      
      else
        {
          // Define lambda function to create unary operator (by clamping kernelParams argument)      
          auto lambda = [=,&p](double d)->double { return (*distKernel)(d, p, deriv); };
          int j;
          #pragma omp parallel for
          for ( j=0 ; j < m; j++ )
            {
              M.col(j) = ((v1.rowwise() - v2.row(j)).rowwise().squaredNorm()).unaryExpr(lambda);          
            }
        }

    }
  else
    {
      // Define lambda function to create unary operator (by clamping kernelParams argument)
      auto lambda = [=,&p](Matrix && x1, Matrix && x2)->double { return (*kernel)(x1, x2, p, deriv); };
      for (auto i : boost::irange(0,n) )
        {
          for (auto j : boost::irange(0,m) )
            {
              M(i,j) = lambda(v1.row(i), v2.row(j));
            }
        }
    }
}


// Compute cross covariance between two input vectors using kernel parameters p
void GP::GaussianProcess::computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p, bool useDistMat=false)
{
  computeCrossCov(M, v1, v2, p, 0, useDistMat);
}

// Compute cross covariance between two input vectors using default kernel parameters
void GP::GaussianProcess::computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, bool useDistMat=false)
{
  computeCrossCov(M, v1, v2, kernelParams, useDistMat);
}


// Compute covariance matrix K with kernel parameters p
void GP::GaussianProcess::computeCov(Vector & p, bool useDistMat=false)
{
  // Check if observations have been defined
  if ( N == 0 ) { std::cout << "[*] No observations (use 'setObs' to define observations).\n"; return; }

  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  K.resize(n,n);

  // Compute cross covariance in place
  computeCrossCov(K, obsX, obsX, p, useDistMat);
}

// Compute covariance matrix K using default kernel paramaters
void GP::GaussianProcess::computeCov(bool useDistMat=false)
{
  computeCov(kernelParams, useDistMat);
}


// Compute Cholesky factorization of covariance matrix K with noise
void GP::GaussianProcess::computeChol(double noise)
{  
  cholesky = ( K + (noise+jitter)*Matrix::Identity(static_cast<int>(K.cols()), static_cast<int>(K.cols())) ).llt();
}


void GP::GaussianProcess::computeCholDirect(double noise, Vector & p)
{
  auto lambda = [=,&p](double d)->double { return (*distKernel)(d, p, 0); };
  auto n = static_cast<int>(obsX.rows());
  cholesky = ( distMatrix.unaryExpr(lambda) + (noise+jitter)*Matrix::Identity(n,n) ).llt();
}
void GP::GaussianProcess::computeCholDirect(Vector & p)
{
  computeCholDirect(noiseLevel, p);
}

// Compute Cholesky factorization of covariance matrix K using default noise
void GP::GaussianProcess::computeChol()
{
  computeChol(noiseLevel);
}
*/

// Evaluate NLML for specified kernel hyperparameters p
// (note: alpha is used to share calculations with DNLML function)
double GP::GaussianProcess::evalNLML(const Vector & p) //, Matrix & alpha)
{
  // Get matrix input observation count
  // Possibly not needed for NLML calculation ... ?
  auto n = static_cast<int>(obsX.rows());

  int paramCount = kernel.getParamCount();
  
  auto pcopy = static_cast<Vector>(p);
  if (!fixedNoise)
    {
      auto noise = static_cast<double>((pcopy.head(1))[0]);
      auto params = static_cast<Vector>( pcopy.tail(paramCount) );        

      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        params(i) = std::exp(params(i));

      //computeCov(params, true);
      //computeCov(params);
      //computeChol(noise);
      //computeCholDirect(noise,params);
      K.resize(n,n);
      kernel.computeCov(K, distMatrix, params, 0);
      cholesky = ( K + (noise+jitter)*Matrix::Identity(n,n) ).llt();
    }
  else
    {
      auto params = static_cast<Vector>( pcopy );

      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        params(i) = std::exp(params(i));

      //computeCov(params, true);
      //computeCov(params);
      //computeChol();
      K.resize(n,n);
      kernel.computeCov(K, distMatrix, params, 0);
      cholesky = ( K + (noiseLevel+jitter)*Matrix::Identity(n,n) ).llt();
    }

  //Matrix cholMat(cholesky.matrixL());
  //Matrix alpha = cholesky.solve(obsY);
  _alpha = cholesky.solve(obsY);

  //auto NLML_value = -(-0.5 * (obsY.transpose()*alpha)(0)  -  cholMat.array().log().matrix().trace() - 0.5*n*std::log(2*PI));
  //auto NLML_value = -(-0.5 * (obsY.transpose()*_alpha)(0)  -  cholMat.array().log().matrix().trace() - 0.5*n*std::log(2*PI));
  //auto NLML_value = -(-0.5 * (obsY.transpose()*_alpha)(0)  -  cholMat.diagonal().array().log().sum() - 0.5*n*std::log(2*PI));
  //auto NLML_value = 0.5*(obsY.transpose()*_alpha)(0)  +  0.5*cholMat.diagonal().array().log().sum() + 0.5*std::log(2*PI);
  auto NLML_value = 0.5*(obsY.transpose()*_alpha)(0)  +  0.5*static_cast<Matrix>(cholesky.matrixL()).diagonal().array().log().sum() + 0.5*std::log(2*PI);
  //std::cout << NLML_value << std::endl;
  return NLML_value;
}


// Evaluate derivatives of NLML for specified kernel hyperparameters
// (note: alpha is used to retrieve calculations from NLML function)
void GP::GaussianProcess::evalDNLML(const Vector & p, Vector & g) //, Matrix & alpha)
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  auto pcopy = static_cast<Vector>(p);

  int paramCount = kernel.getParamCount();
  
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
    //g[0] = -0.5 * ( alpha*alpha.transpose() - cholesky.solve(Matrix::Identity(n,n)) ).trace() ;
    //g[0] = -0.5 * ( alpha*alpha.transpose()*noiseLevel - cholesky.solve(noiseLevel*Matrix::Identity(n,n)) ).trace() ;
  else
    shift = 1;

  Matrix term = cholesky.solve(Matrix::Identity(n,n)) - _alpha*_alpha.transpose();
  Matrix dK_i(n,n);
  double trace;
  for (auto i : boost::irange(1,paramCount+1))
    {
      //computeCrossCov(dK_i, obsX, obsX, params, i);
      //g[i-shift] = -0.5 * ( alpha*alpha.transpose()*dK_i - cholesky.solve(dK_i) ).trace() ;
      // g[i-shift] = -0.5 * ( _alpha*_alpha.transpose()*dK_i - cholesky.solve(dK_i) ).trace() ;
      //g[i-shift] = 0.5 * ((cholesky.solve(Matrix::Identity(n,n)) - _alpha*_alpha.transpose())*dK_i ).trace() ;
      
      //computeCrossCov(dK_i, obsX, obsX, params, i, true);
      kernel.computeCov(dK_i, distMatrix, params, i);
      //g[i-shift] = 0.5 * (term * dK_i ).trace() ;
      trace = 0.0;
      for (auto j : boost::irange(0,n))
        {
          trace += term.row(j)*dK_i.col(j);
        }
      
      g[i-shift] = 0.5*trace;
    }
}


// Fit model hyperparameters
void GP::GaussianProcess::fitModel()
{
  // Get combined parameter/noise vector size
  int paramCount = kernel.getParamCount();
  int n = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

  // Precompute distance matrix
  computeDistMat();
  
  // Specify initial parameter values for solver
  Vector x = static_cast<Vector>(Eigen::VectorXd::Ones(n));
  if (!fixedNoise)
    x[0] = 0.001;
  
  // Declare vector for storing gradient calculations
  Vector D(n);

  // Specify precision of minimization algorithm
  double SIG = 0.1;
  double EXT = 3.0;
  //double INT = 0.01;
  //int MAX = 20;
  //int length = 1000;
  //double INT = 0.1;
  //int MAX = 10;
  int MAX = 40;
  double INT = 0.01;
  int length = 1000;

  // Optimize hyperparameters
  //minimize::cg_minimize(x, this , D, length, SIG, EXT, INT);
  minimize::cg_minimize(x, this , D, length, SIG, EXT, INT, MAX);

  // Assign tuned parameters to model
  if (!fixedNoise)
    {
      noiseLevel = static_cast<double>((x.head(1))[0]);
      Vector optParams = x.tail(x.size() - 1);
      
      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        optParams(i) = std::exp(optParams(i));

      kernel.setParams(optParams);
    }
  else
    {
      Vector optParams = x;

      // ASSUME OPTIMIZATION OVER LOG VALUES
      for ( auto i : boost::irange(0,paramCount) )
        optParams(i) = std::exp(optParams(i));

      kernel.setParams(optParams);
    }


  // Recompute covariance and Cholesky factor
  //computeCov();
  //computeChol();
};

/*
// Compute predicted values
void GP::GaussianProcess::predict()
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  auto m = static_cast<int>(predX.rows());

  // Store cross covariance for test points in kstar
  Matrix kstar;
  kstar.resize(n,m);
  computeCrossCov(kstar, obsX, predX);

  Matrix kstarmat;
  kstarmat.resize(m,m);
  computeCrossCov(kstarmat, predX, predX);
  
  Matrix cholMat(cholesky.matrixL());
  Matrix alpha = cholesky.solve(obsY);
  Matrix v = kstar;
  cholMat.triangularView<Eigen::Lower>().solveInPlace(v);

  // Verify that  L*v = kstar  and  (K+noise)*alpha = obsY
  //std::cout << (cholMat.triangularView<Eigen::Lower>() * v - kstar).squaredNorm() << std::endl;
  //auto noiseK = K + (noiseLevel+jitter)*Matrix::Identity(static_cast<int>(K.cols()), static_cast<int>(K.cols()));
  //std::cout << (noiseK * alpha - obsY).squaredNorm() << std::endl;
  
  // Set predictive means/variances and compute negative log marginal likelihood
  predMean = kstar.transpose() * alpha;
  predCov = kstarmat - v.transpose() * v;
  //NLML = -0.5 * (obsY.transpose()*alpha)(0)  -  cholMat.array().log().matrix().trace() - 0.5*n*std::log(2*PI);
  
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
*/
