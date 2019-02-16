#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <memory>
#include <random>
#include <limits>
#include <boost/range/irange.hpp>
#include <Eigen/Dense>
#include "GPs.h"

// Include CppOptLib files
#include "./include/cppoptlib/meta.h"
#include "./include/cppoptlib/boundedproblem.h"
#include "./include/cppoptlib/solver/lbfgsbsolver.h"

// Retrieve aliases from GP namescope
using Matrix = GP::Matrix;
using Vector = GP::Vector;


// Compute pairwise distance between lists of points
void GP::pdist(Matrix & Dv, Matrix & X1, Matrix & X2)
{
  auto n = static_cast<int>(X1.rows());
  int k = 0;
  auto entryCount = static_cast<int>( (n*(n-1))/2);
  Dv.resize(entryCount, 1);
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

  int k = 0;
  for ( auto i : boost::irange(0,n-1) )
    {
      for ( auto j : boost::irange(i+1,n) )
        D(i,j) = D(j,i) = Dv(k++,0);
    }

  D.diagonal() = diagVal * Eigen::MatrixXd::Ones(n,1);
}


// Compute covariance matrix (and gradients) from a vector of squared pairwise distances Dv
std::vector<Matrix> GP::RBF::computeCov(Matrix & K, Matrix & Dv, Vector & params, double jitter, bool evalGrad)
{
  auto n = static_cast<int>(K.rows());
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

  // Compute gradient list if "evalGrad=true"
  std::vector<Matrix> gradList;
  if ( evalGrad )
    {
      dK_iv.noalias() = 1/std::pow(params(lengthIndex),2) * ( Dv.array() * Kv.array() ).matrix();
      squareForm(dK_i, dK_iv, n); // Note: diagVal = 0.0
      gradList.push_back(dK_i);
    }

  return gradList;
  
};


// Compute cross covariance between two input vectors using kernel parameters params
void GP::RBF::computeCrossCov(Matrix & K, Matrix & X1, Matrix & X2, Vector & params)
{
  // Get prediction count
  auto m = static_cast<int>(X2.rows());

   // Define lambda function to create unary operator (by clamping kernelParams argument)      
  auto lambda = [=,&params](double d)->double { return evalDistKernel(d, params, 0); };
  for ( auto j : boost::irange(0,m) )
    {
      K.col(j) = ((X1.rowwise() - X2.row(j)).rowwise().squaredNorm()).unaryExpr(lambda);          
    }
  
};

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
// [ Note: Optimize w.r.t. theta = log(l) for stability  ==>   / l^2  instead of  / l^3 ]
double GP::RBF::evalDistKernel(double d, Vector & params, int n)
{
  switch (n)
    {
    case 0: return std::exp( -d / (2.0*std::pow(params(0),2)));
    case 1: return d / std::pow(params(0),2) * std::exp( -d / (2.0*std::pow(params(0),2)));
    default: std::cout << "\n[*] UNDEFINED DERIVATIVE\n"; return 0.0;
    }
};


// Precompute distance matrix to avoid repeated calculations during optimization procedure
void GP::GaussianProcess::computeDistMat()
{
  pdist(distMatrix, obsX, obsX);
}


// Evaluate NLML for specified kernel hyperparameters p
double GP::GaussianProcess::evalNLML(const Vector & p, Vector & g, bool evalGrad)
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());

  // ASSUME OPTIMIZATION OVER LOG VALUES
  auto params = static_cast<Vector>(p);
  params = params.array().exp().matrix();

  // Compute covariance matrix and store Cholesky factor
  K.resize(n,n);
  time start = high_resolution_clock::now();
  auto gradList = (*kernel).computeCov(K, distMatrix, params, jitter, evalGrad);
  time end = high_resolution_clock::now();
  time_computecov += getTime(start, end);

  start = high_resolution_clock::now();
  cholesky = K.llt();
  end = high_resolution_clock::now();
  time_cholesky_llt += getTime(start, end);

  // Store alpha for DNLML calculation
  _alpha.noalias() = cholesky.solve(obsY);

  // Compute NLML value
  start = high_resolution_clock::now();
  double NLML_value = 0.5*(obsY.transpose()*_alpha)(0);
  NLML_value += static_cast<Matrix>(cholesky.matrixL()).diagonal().array().log().sum();
  NLML_value += 0.5*n*std::log(2*PI);
  end = high_resolution_clock::now();
  time_NLML += getTime(start, end);


  if ( evalGrad )
    {

      //
      // Precompute the multiplicative term in derivative expressions
      //
      // [ THIS APPEARS TO BE A COMPUTATIONAL BOTTLE-NECK ]
      //

      start = high_resolution_clock::now();
      // Direct evaluation of inverse matrix
      //term.noalias() = cholesky.solve(Matrix::Identity(n,n)) - _alpha*_alpha.transpose();

      // Try forcing Eigen to solve in place
      term.noalias() = Matrix::Identity(n,n);
      cholesky.solveInPlace(term);
      term.noalias() -= _alpha*_alpha.transpose();
      
      /*
      //
      //          ---  PARALLEL IMPLEMENTATION  ---
      //
      //  [ Typically much slower; possibly not thread-safe... ]
      //
      term.resize(n,n);
      //Matrix ei = Eigen::MatrixXd::Zero(n,1);
      int i = 0;
      //#pragma omp parallel for private(i) shared(cholesky, term, n)
#pragma omp parallel for private(i) shared(cholesky, _alpha, term, n)
      for ( i=0 ; i<n ; i++ )
        {
          Matrix ei = Eigen::MatrixXd::Zero(n,1);
          ei(i) = 1.0;
          //term.col(i) = cholesky.solve(ei);
          term.col(i) = cholesky.solve(ei);
          term.col(i) -= _alpha(i)*_alpha;
        }
      */

      
      //term.noalias() -= _alpha*_alpha.transpose();
      end = high_resolution_clock::now();
      time_term += getTime(start, end);

      
      start = high_resolution_clock::now();      
      // Compute gradient for noise term if 'fixedNoise=false'
      int index = 0;
      if (!fixedNoise)
        {
          // Specify gradient of white noise kernel  [ dK_i = params(0)*Matrix::Identity(n,n) ]
          g(index++) = 0.5 * (term * params(0)).trace() ;
        }

      // Compute gradients with respect to kernel hyperparameters
      for (auto dK_i = gradList.begin(); dK_i != gradList.end(); ++dK_i) 
        {
          // Compute trace of full matrix
          g(index++) = 0.5 * (term * (*dK_i) ).trace() ;
        }
      end = high_resolution_clock::now();
      time_grad += getTime(start, end);

      // Update gradient evaluation count
      gradientEvals += 1;

    }

  return NLML_value;
  
}


// Define simplified interface for evaluating NLML without gradient calculation
double GP::GaussianProcess::evalNLML(const Vector & p)
{
  Vector nullGrad(0);
  return evalNLML(p,nullGrad,false);
}


// Define utility function for formatting hyperparameter bounds
void GP::GaussianProcess::parseBounds(Vector & lbs, Vector & ubs, int augParamCount)
{
  lbs.resize(augParamCount);
  ubs.resize(augParamCount);

  double defaultLowerBound = 0.00001;
  double defaultUpperBound = 10.0;
  
  if ( fixedBounds )
    {
      // Check if bounds for noise parameter were provided
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

}


// Fit model hyperparameters
void GP::GaussianProcess::fitModel()
{

  // Get combined parameter/noise vector size
  paramCount = (*kernel).getParamCount();
  augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;

  // Pass noise level to kernel when 'fixedNoise=true'
  if ( fixedNoise )
    (*kernel).setNoise(noiseLevel);
  
  // Precompute distance matrix
  computeDistMat();

  // Declare vector for storing gradient calculations
  Vector g(augParamCount);

  // Initialize gradient vector size
  cppOptLibgrad.resize(augParamCount);

  // Convert hyperparameter bounds to log-scale
  Vector lbs, ubs;
  parseBounds(lbs, ubs, augParamCount);

  Vector optParams = Eigen::MatrixXd::Zero(augParamCount,1);
  
  this->setLowerBound(lbs);
  this->setUpperBound(ubs);
  cppoptlib::LbfgsbSolver<GaussianProcess> solver;

  // Specify stopping criteria
  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  //crit.iterations = 10; //!< Maximum number of iterations
  crit.gradNorm = 1e-6;   //!< Minimum norm of gradient vector
  //crit.fDelta = 7.5e-5;   //!< Minimum [relative] change in cost function
  crit.iterations = solverIterations;
  crit.fDelta = solverPrecision;
  solver.setStopCriteria(crit);
  solver.setHistorySize(10);
  
  solver.minimize(*this, optParams);

  // Display final solver criteria values
  std::cout << "\nSolver Criteria |";
  std::cout << "\n----------------\n" << solver.criteria() << std::endl;
  std::cout << "gradEvals =\t" << gradientEvals <<std::endl;

  
  // ASSUME OPTIMIZATION OVER LOG VALUES
  optParams = optParams.array().exp().matrix();


  ///* [ This is included in the SciKit Learn model.fit() call as well ]

  // Recompute covariance and Cholesky factor
  auto nullGradList = (*kernel).computeCov(K, distMatrix, optParams);
  cholesky = K.llt();
  _alpha.noalias() = cholesky.solve(obsY);

  
  // Assign tuned parameters to model
  if (!fixedNoise)
    {
      noiseLevel = static_cast<double>((optParams.head(1))[0]);
      optParams = static_cast<Vector>(optParams.tail(augParamCount - 1));
    }
  
  (*kernel).setParams(optParams);


  // DISPLAY TIMING INFORMATION
  /*
  std::cout << "\n Time Diagnostics |\n";
  std::cout << "------------------\n";
  std::cout << "computeCov():\t  " << time_computecov  << std::endl;
  std::cout << "cholesky.llt():\t  " << time_cholesky_llt  << std::endl;
  std::cout << "NLML:\t  \t  " << time_NLML  << std::endl;
  std::cout << "Grad term:\t  " << time_term  << std::endl;
  std::cout << "Gradient:\t  " << time_grad  << std::endl;
  */
  
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
  Matrix kstar_and_v;
  kstar_and_v.resize(n,m);
  (*kernel).computeCrossCov(kstar_and_v, obsX, predX, params);

  // Compute covariance matrix for test points
  Matrix kstarmat;
  kstarmat.resize(m,m);
  (*kernel).computeCrossCov(kstarmat, predX, predX, params);

  // Set predictive means/variances and compute negative log marginal likelihood
  Matrix cholMat(cholesky.matrixL());
  predMean.noalias() = kstar_and_v.transpose() * _alpha;
  cholMat.triangularView<Eigen::Lower>().solveInPlace(kstar_and_v);  // kstar_and_v is now 'v'
  predCov.noalias() = kstarmat - kstar_and_v.transpose() * kstar_and_v;

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


// Evaluate NLML [public interface]
double GP::GaussianProcess::computeNLML(const Vector & p, double noise)
{
  // Compute log-hyperparameters
  Vector logparams(augParamCount);

  if ( !fixedNoise )
    {
      logparams(0) = std::log(noise);
      for ( auto i : boost::irange(1,augParamCount) )
        logparams(i) = std::log(p(i-1));
    }
  else
    {
      logparams = logparams.array().log().matrix();
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



// Define function for uniform sampling
Matrix GP::sampleUnif(double a, double b, int N, int dim)
{
  //return (b-a)*(Eigen::MatrixXd::Random(N,1) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,1)) + a*Eigen::MatrixXd::Ones(N,1);
  return (b-a)*(Eigen::MatrixXd::Random(N,dim) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,dim) ) + a*Eigen::MatrixXd::Ones(N,dim);
}


// Define function for uniform sampling [Vectors]
Vector GP::sampleUnifVector(Vector lbs, Vector ubs)
{
  auto n = static_cast<int>(lbs.rows());

  Vector sampleVector = ((ubs-lbs).array()*( 0.5*Eigen::MatrixXd::Random(n,1) + 0.5*Eigen::MatrixXd::Ones(n,1) ).array()).matrix() + (lbs.array()*Eigen::MatrixXd::Ones(n,1).array()).matrix();
  
  return sampleVector;
}


// Define function for sampling from standard normal distribution
Matrix GP::sampleNormal(int N)
{
  // Note: Boost random is currently throwing deprecated header warnings...
  //boost::random::mt19937 rng;
  //boost::random::normal_distribution<> normalDist;

  /*
  // [ NOTE: .noalias() is 100% necessary with "std::default_random_engine" ]
  std::default_random_engine rng;
  std::normal_distribution<double> normalDist(0.0,1.0);
  Matrix sampleVals(N,1);
  for ( auto i : boost::irange(0,N) )
    sampleVals(i) = normalDist(rng);
  */

  
  //
  //          Crude implementation of Box-Muller Transform
  // ( see https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform )
  //
  static const double epsilon = std::numeric_limits<double>::min();
  Matrix U1(N,1);
  Matrix U2(N,1);

  // Ensure log operand is not too small
  double scaledLimit = 2.0*epsilon - 1.0;
  do
    {
      U1 = Eigen::MatrixXd::Random(N,1);
      U2 = Eigen::MatrixXd::Random(N,1);
    }
  while ( U1.minCoeff() <= scaledLimit );

  // Rescale and shift Unif(-1,1) values to the interval [0,1]
  U1 =  0.5*(U1 + Eigen::MatrixXd::Ones(N,1));
  U2 =  0.5*(U2 + Eigen::MatrixXd::Ones(N,1));

  // Compute transformed values with .noalias() to ensure evaluation
  Matrix sampleVals;
  sampleVals.noalias() = ((-2.0*(U1.array().log()).matrix()).array().sqrt() * (2*PI*U2).array().cos()).matrix();

  return sampleVals;
}


// Generate equally spaced points on an interval or square region
Matrix GP::linspace(double a, double b, int N, int dim)
{

  Matrix linspaceVals;
  if ( dim == 1 )
    {
      linspaceVals.resize(N,1);
      linspaceVals = Eigen::Array<double, Eigen::Dynamic, 1>::LinSpaced(N, a, b);
    }
  else if ( dim == 2 )
    {
      linspaceVals.resize(N*N,2);
      Matrix linspaceVals1D = Eigen::Array<double, Eigen::Dynamic, 1>::LinSpaced(N, a, b);
      int k = 0;
      for ( auto i : boost::irange(0,N) )
        {
          for ( auto j : boost::irange(0,N) )
            {
              linspaceVals(k,0) = linspaceVals1D(i);
              linspaceVals(k,1) = linspaceVals1D(j);
              k++;
            }
        }
    }
  else
      std::cout << "[*] GP::linspace has not been implemented for dim > 2\n";
  
  return linspaceVals;
}

// Define function for retrieving time from chrono
float GP::getTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
  return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);
};






/*
// POTENTIAL PARALLEL IMPLEMENTATION OF SQUARE FORM; SPEED-UP APPEARS NEGLIGIBLE
// Re-assemble pairwise distances into a dense matrix
void GP::squareFormParallel(Matrix & D, Matrix & Dv, int n, double diagVal)
{

  D.resize(n,n);
  int i;
  int j;
#pragma omp parallel for private(i,j) shared(D,Dv,n)
  for ( i = 0 ; i<n-1; i++ )
    {
      for ( j = i+1 ; j<n ; j++ )
        {
          D(i,j) = D(j,i) = Dv( static_cast<int>(i*n - (i*(i+1))/2 + j - i -1) ,0);
        }
    }
  D.diagonal() = diagVal * Eigen::MatrixXd::Ones(n,1);
}
*/




//
//   ORIGINAL MINIMIZATION IMPLEMENTATION USING RASMUSSEN'S CODE
//

//
//    NOTE:
//
//    Remember to include "utils/minimize.h" and derive the
//   'GaussianProcess' class from the 'GradientObj' class:
//
//    i.e.  class GaussianProcess : public minimize::GradientObj
//

/*
void minimize(...)
{


  //
  // Specify the parameters for the minimization algorithm
  //
  
  //   HIGH ACCURACY SETTINGS   //
  // max of MAX function evaluations per line search
  //int MAX = 30;
  // max number of line searches = length
  //int length = 20;
  // don't reevaluate within INT of the limit of the current bracket
  //double INT = 0.00001;
  // SIG is a constant controlling the Wolfe-Powell conditions
  //double SIG = 0.9;
  // extrapolate maximum EXT times the current step-size
  //double EXT = 5.0;

  //  EFFICIENT SETTINGS  //


  int MAX = 15;
  int length = 10;
  double INT = 0.00001;
  double SIG = 0.9;
  double EXT = 5.0;

  // Define number of exploratory NLML evaluations for specifying
  // a reasonable initial value for the optimization algorithm
  int initParamSearchCount = 30;
    
  // Define restart count for optimizer
  int restartCount = 0;
  
  // Convert hyperparameter bounds to log-scale
  Vector lbs, ubs;
  parseBounds(lbs, ubs, augParamCount);

  // Declare variables to store optimization loop results
  double currentVal;
  double optVal = 1e9;
  Vector theta(augParamCount);
  Vector optParams(augParamCount);

  //time start = high_resolution_clock::now();
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
  //time end = high_resolution_clock::now();
  //time_paramsearch += getTime(start, end);


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
  //start = high_resolution_clock::now();
  minimize::cg_minimize(optParams, this, g, length, SIG, EXT, INT, MAX);
  //end = high_resolution_clock::now();
  //time_minimize += getTime(start, end);

}
*/
