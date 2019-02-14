//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <memory>
#include <random>
#include <boost/range/irange.hpp>
#include <Eigen/Dense>
#include "GPs.h"

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


// Compute pairwise distance between lists of points
void GP::pdist(Matrix & Dv, Matrix & X1, Matrix & X2)
{
  auto n = static_cast<int>(X1.rows());
  int k = 0;
  auto entryCount = static_cast<int>( (n*(n-1))/2);
  Dv.resize(entryCount, 1);
  //#pragma omp parallel for 
  for ( auto i : boost::irange(0,n-1) )
    {      
      for ( auto j : boost::irange(i+1,n) )
          Dv(k++) = (static_cast<Vector>(X1.row(i) - X2.row(j))).squaredNorm();
    }
}

// Re-assemble pairwise distances into a dense matrix
void GP::squareForm(Matrix & D, Matrix & Dv, int n, double diagVal)
{

  D.resize(n,n);
  /*
  // POTENTIAL PARALLEL IMPLEMENTATION; SPEED-UP APPEARS NEGLIGIBLE
  //int k;
  //int shift;
  int i;
  int j;
  //#pragma omp parallel for private(i,j,k,shift) shared(D,Dv,n)
#pragma omp parallel for private(i,j) shared(D,Dv,n)
  for ( i = 0 ; i<n-1; i++ )
    {
      for ( j = i+1 ; j<n ; j++ )
        {
          //shift = i+1;
          //k = static_cast<int>(i*n - (i*(i+1))/2 + j - shift);
          //D(i,j) = D(j,i) = Dv( k ,0);
          D(i,j) = D(j,i) = Dv( static_cast<int>(i*n - (i*(i+1))/2 + j - i -1) ,0);
        }
    }
  */
  
  ///*
  int k = 0;
  for ( auto i : boost::irange(0,n-1) )
    {
      for ( auto j : boost::irange(i+1,n) )
        D(i,j) = D(j,i) = Dv(k++,0);
    }
  //*/

  D.diagonal() = diagVal * Eigen::MatrixXd::Ones(n,1);
}

// Compute cross covariance between two input vectors using kernel parameters p
std::vector<Matrix> GP::RBF::computeCov(Matrix & K, Matrix & Dv, Vector & params, double jitter, bool evalGrad)
{
  auto n = static_cast<int>(K.rows());
  
  // Define lambda function to create unary operator (by clamping kernelParams argument)      
  //auto lambda = [=,&params](double d)->double { return evalDistKernel(d, params, 0); };

  //double diagVal = 1.0;
  //Matrix Kv = Dv.unaryExpr(lambda);
  //Kv = Dv.unaryExpr(lambda);
  //squareForm(K, Kv, n, diagVal);
  int lengthIndex;
  if ( params.size() > paramCount )
    {
      Kv.noalias() = ( (-0.5 / std::pow(params(1),2)) * Dv ).array().exp().matrix();
      squareForm(K, Kv, n, 1.0 + params(0) + jitter);
      lengthIndex = 1;
    }
  else
    {
      Kv.noalias() = ( (-0.5 / std::pow(params(0),2)) * Dv ).array().exp().matrix();
      squareForm(K, Kv, n, 1.0 + noiseLevel + jitter);
      lengthIndex = 0;
    }

  // Original formulation:
  //Matrix D;
  //squareForm(D, Dv, n);
  //K.noalias() = D.unaryExpr(lambda);

  // Compute gradient list if "evalGrad=true"
  std::vector<Matrix> gradList;
  if ( evalGrad )
    {
      //Matrix dK_i(n,n);
      //Matrix dK_iv = ( Dv.array() * (1/std::pow(params(0),2)) * Kv.array() ).matrix();
      //dK_iv.noalias() = 1/std::pow(params(0),2) * ( Dv.array() * Kv.array() ).matrix();
      dK_iv.noalias() = 1/std::pow(params(lengthIndex),2) * ( Dv.array() * Kv.array() ).matrix();
      squareForm(dK_i, dK_iv, n); // Note: diagVal = 0.0

      // Original formulation:
      //Matrix dK_i = ( D.array() * (1/std::pow(params(0),2)) * K.array() ).matrix();
      
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
  //auto n = static_cast<int>(obsX.rows());
  /*
  distMatrix.resize(n,n);  
  for ( int j=0 ; j < n; j++ )
    {
      distMatrix.col(j) = (obsX.rowwise() - obsX.row(j)).rowwise().squaredNorm();
    }
  */
  pdist(distMatrix, obsX, obsX);
  
}

// Evaluate NLML [public interface]
double GP::GaussianProcess::computeNLML(const Vector & p, double noise)
{
  // Get hyperparameter count ( + noise parameter )
  //int paramCount = (*kernel).getParamCount();
  //int augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

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
      logparams = logparams.array().log().matrix();
      //for ( auto i : boost::irange(0,paramCount) )
      //  logparams(i) = std::log(p(i));
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
  //int paramCount = (*kernel).getParamCount();
  //int augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

  // ASSUME OPTIMIZATION OVER LOG VALUES
  auto params = static_cast<Vector>(p);
  params = params.array().exp().matrix();
  //for ( auto i : boost::irange(0,augParamCount) )
  //  pcopy(i) = std::exp(pcopy(i));

  /*
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
  */

  // Compute covariance matrix and store Cholesky factor
  K.resize(n,n);
  //auto gradList = (*kernel).computeCov(K, distMatrix, params, jitter, evalGrad);
  auto gradList = (*kernel).computeCov(K, distMatrix, params, jitter, evalGrad);
  //cholesky = ( K + (noise+jitter)*Matrix::Identity(n,n) ).llt();
  cholesky = K.llt();
  //cholesky = ( (K + (noise+jitter)*Matrix::Identity(n,n)).eval() ).llt();

  // Store alpha for DNLML calculation
  //_alpha = cholesky.solve(obsY);
  _alpha.noalias() = cholesky.solve(obsY);

  // Compute NLML value
  //auto NLML_value = 0.5*(obsY.transpose()*_alpha)(0) + static_cast<Matrix>(cholesky.matrixL()).diagonal().array().log().sum() + 0.5*n*std::log(2*PI);
  double NLML_value = 0.5*(obsY.transpose()*_alpha)(0);
  NLML_value += static_cast<Matrix>(cholesky.matrixL()).diagonal().array().log().sum();
  NLML_value += 0.5*n*std::log(2*PI);

  if ( evalGrad )
    {

      // Precompute the multiplicative term in derivative expressions
      //Matrix term = cholesky.solve(Matrix::Identity(n,n)) - _alpha*_alpha.transpose();

      // THIS APPEARS TO BE A COMPUTATIONAL BOTTLE-NECK
      term.noalias() = cholesky.solve(Matrix::Identity(n,n)) - _alpha*_alpha.transpose();

      
      // Compute gradient for noise term if 'fixedNoise=false'
      int index = 0;
      if (!fixedNoise)
        {
          // Specify gradient of white noise kernel
          //Matrix dK_noise = noise*Matrix::Identity(n,n);
          
          // Compute trace of full matrix
          //g(index) = 0.5 * (term * dK_noise ).trace() ;
          //g(index) = 0.5 * (term * noise).trace() ;
          g(index++) = 0.5 * (term * params(0)).trace() ;
        }

      // Compute gradients with respect to kernel hyperparameters
      for (auto dK_i = gradList.begin(); dK_i != gradList.end(); ++dK_i) 
        {
          // Compute trace of full matrix
          g(index++) = 0.5 * (term * (*dK_i) ).trace() ;
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
  //return (b-a)*(Eigen::MatrixXd::Random(N,1) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,1)) + a*Eigen::MatrixXd::Ones(N,1);
  return (b-a)*(Eigen::MatrixXd::Random(N,1) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,1) ) + a*Eigen::MatrixXd::Ones(N,1);
}

// Define function for uniform sampling [Vectors]
Vector GP::sampleUnifVector(Vector lbs, Vector ubs)
{

  auto n = static_cast<int>(lbs.rows());
  //Vector samples = Eigen::MatrixXd::Random(n,1);
  Vector sampleVector = ((ubs-lbs).array()*( 0.5*Eigen::MatrixXd::Random(n,1) + 0.5*Eigen::MatrixXd::Ones(n,1) ).array()).matrix() + (lbs.array()*Eigen::MatrixXd::Ones(n,1).array()).matrix();

  /*
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
      //sampleVector(i) = (b-a)*(samples(i) * 0.5 + 0.5*samples(i)) + a*samples(i);
      sampleVector(i) = (b-a)*(samples(i) * 0.5 + 0.5) + a;
    }
  */
  return sampleVector;
}


// Define utility function for formatting hyperparameter bounds
void GP::GaussianProcess::parseBounds(Vector & lbs, Vector & ubs, int augParamCount)
{
  lbs.resize(augParamCount);
  ubs.resize(augParamCount);

  double defaultLowerBound = 0.00001;
  //double defaultUpperBound = 10.0;
  double defaultUpperBound = 5.0;
  
  if ( fixedBounds )
    {
      // Check if bounds for noise parameter were provided
      //auto bsize = static_cast<int>(lowerBounds.rows());
      //if ( bsize < augParamCount )
      if ( lowerBounds.size() < augParamCount )
        {
          // Set noise bounds to defaults
          lbs(0) = std::log( defaultLowerBound );
          ubs(0) = std::log( defaultUpperBound );

          // Convert specified bounds to log-scale
          for ( auto bi : boost::irange(1,augParamCount) )
            {
              lbs(bi) = std::log(lowerBounds(bi-1));
              ubs(bi) = std::log(upperBounds(bi-1));
            }
        }
      else
        {
          // Convert specified bounds to log-scale
          lbs = (lowerBounds.array().log()).matrix();
          ubs = (upperBounds.array().log()).matrix();
        }
    }
  else
    {
      // Set noise and hyperparameter bounds to defaults
      lbs = ( defaultLowerBound * Eigen::MatrixXd::Ones(augParamCount,1) ).array().log().matrix();
      ubs = ( defaultUpperBound * Eigen::MatrixXd::Ones(augParamCount,1) ).array().log().matrix();
    }

  /*
  std::cout << "Bounds:\n";
  std::cout << lbs.array().exp().matrix().transpose() << std::endl;
  std::cout << ubs.array().exp().matrix().transpose() << std::endl;
  std::cout << "\nLog Bounds:\n";
  std::cout << lbs.transpose() << std::endl;
  std::cout << ubs.transpose() << std::endl << std::endl;
  */

}


// Fit model hyperparameters
void GP::GaussianProcess::fitModel()
{
  // Get combined parameter/noise vector size
  //int paramCount = (*kernel).getParamCount();
  //int augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;
  // Get combined parameter/noise vector size
  paramCount = (*kernel).getParamCount();
  augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;


  if ( fixedNoise )
    (*kernel).setNoise(noiseLevel);
  
  // Precompute distance matrix
  computeDistMat();

  // Declare vector for storing gradient calculations
  Vector g(augParamCount);

  //
  // Specify the parameters for the minimization algorithm
  //
  
  // /*  HIGH ACCURACY SETTINGS  */ //
  /*
  // max of MAX function evaluations per line search
  int MAX = 30;
  // max number of line searches = length
  int length = 20;
  // don't reevaluate within INT of the limit of the current bracket
  double INT = 0.00001;
  // SIG is a constant controlling the Wolfe-Powell conditions
  double SIG = 0.9;
  // extrapolate maximum EXT times the current step-size
  double EXT = 5.0;
  */

  // /* EFFICIENT SETTINGS */ //
  int MAX = 15;
  int length = 10;
  double INT = 0.00001;
  double SIG = 0.9;
  double EXT = 5.0;

  // Define number of exploratory NLML evaluations for specifying
  // a reasonable initial value for the optimization algorithm
  int initParamSearchCount = 30;
    
  // Define restart count for optimizer
  //int restartCount = 10;
  int restartCount = 0;
  
  // Convert hyperparameter bounds to log-scale
  Vector lbs, ubs;
  parseBounds(lbs, ubs, augParamCount);

  // Declare variables to store optimization loop results
  double currentVal;
  double optVal = 1e9;
  Vector theta(augParamCount);
  Vector optParams(augParamCount);


  // First explore hyperparameter space to get a reasonable initializer for optimization
  for ( auto i : boost::irange(0,initParamSearchCount) )
    {
      if ( i == 0 )
          theta = Eigen::MatrixXd::Zero(augParamCount,1);
      else
          theta = sampleUnifVector(lbs, ubs);

      // Compute current NLML and store parameters if optimal
      currentVal = evalNLML(theta);
      if ( currentVal < optVal )
        {
          optVal = currentVal;
          optParams = theta;
        }
      
      //std::cout << "Theta Search:  " << theta.transpose() << "  [ NLML = " << currentVal << " ]"<< std::endl;

    }

  // NOTE: THIS NEEDS TO BE RE-WRITTEN TO USE THE PRELIMINARY PARAMETER SEARCH RESULTS
  // Evaluate optimizer with various different initializations
  for ( auto i : boost::irange(0,restartCount) )
    {
      // Avoid compiler warning for unused variable
      //(void)i;

      if ( i == 0 )
        {
          // Set initial guess (should make this user specifiable...)
          theta = Eigen::MatrixXd::Zero(augParamCount,1);
        }
      else
        {
          // Sample initial hyperparameter vector
          theta = sampleUnifVector(lbs, ubs);
        }

      // Optimize hyperparameters
      minimize::cg_minimize(theta, this, g, length, SIG, EXT, INT, MAX);
      
      // Compute current NLML and store parameters if optimal
      currentVal = evalNLML(theta);
      if ( currentVal < optVal )
        {
          optVal = currentVal;
          optParams = theta;
        }
    }

  // Perform one last optimization starting from best parameters so far
  if ( ( initParamSearchCount == 0 ) && ( restartCount == 0 ) )
    optParams = sampleUnifVector(lbs, ubs);
  //std::cout << "\n[*] FINAL - Initial Values (log):  " << optParams.transpose() << std::endl;
  //std::cout << "[*] FINAL - Initial Values (std):  " << optParams.transpose().array().exp().matrix() << std::endl;
  minimize::cg_minimize(optParams, this, g, length, SIG, EXT, INT, MAX);
  
  // ASSUME OPTIMIZATION OVER LOG VALUES
  optParams = optParams.array().exp().matrix();
  //for ( auto i : boost::irange(0,augParamCount) )
  //  optParams(i) = std::exp(optParams(i));
  
  // Assign tuned parameters to model
  if (!fixedNoise)
    {
      noiseLevel = static_cast<double>((optParams.head(1))[0]);
      optParams = static_cast<Vector>(optParams.tail(augParamCount - 1));
    }
  
  (*kernel).setParams(optParams);

  ///* [May be able to omit this if last NLML evaluation was optimal]
  // Recompute covariance and Cholesky factor
  auto N = static_cast<int>(K.rows());
  Vector params = (*kernel).getParams();
  auto nullGradList = (*kernel).computeCov(K, distMatrix, params);
  cholesky = ( K + (noiseLevel+jitter)*Matrix::Identity(N,N) ).llt();
  //*/

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

