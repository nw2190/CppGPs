#include <iostream>
#include <cmath>
#include <array>
#include <chrono>
#include <random>
#include <boost/range/irange.hpp>
#include <Eigen/Dense>
#include "GPs.h"

//using namespace GP;
//using Matrix = Eigen::MatrixXd;
//using Vector = Eigen::VectorXd;
using Matrix = GP::Matrix;
using Vector = GP::Vector;

// Compute cross covariance between two input vectors using kernel parameters p
// (note: values of deriv > 0 correspond to derivative calculations)
void GP::GaussianProcess::computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p, int deriv)
{
  // Get matrix input observation count
  auto n = static_cast<int>(v1.rows());
  auto m = static_cast<int>(v2.rows());

  // Evaluate using vectorized implementation of distance kernel,
  // or apply standard binary kernel function component-wise
  if (useDistKernel)
    {
      /*
      // Define lambda function to create unary operator (by clamping kernelParams argument)      
      auto lambda = [=,&p](double d)->double { return (*distKernel)(d, p, deriv); };
      for (auto j : boost::irange(0,m) )
        {
          M.col(j) = ((v1.rowwise() - v2.row(j)).rowwise().squaredNorm()).unaryExpr(lambda);          
        }
      */

      // Define lambda function to create unary operator (by clamping kernelParams argument)      
      auto lambda = [=,&p](double d)->double { return (*distKernel)(d, p, deriv); };
      int j;
      #pragma omp parallel for
      for ( j=0 ; j < m; j++ )
        {
          M.col(j) = ((v1.rowwise() - v2.row(j)).rowwise().squaredNorm()).unaryExpr(lambda);          
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
void GP::GaussianProcess::computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p)
{
  computeCrossCov(M, v1, v2, p, 0);
}

// Compute cross covariance between two input vectors using default kernel parameters
void GP::GaussianProcess::computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2)
{
  computeCrossCov(M, v1, v2, kernelParams);
}


// Compute covariance matrix K with kernel parameters p
void GP::GaussianProcess::computeCov(Vector & p)
{
  // Check if observations have been defined
  if ( N == 0 ) { std::cout << "[*] No observations (use 'setObs' to define observations).\n"; return; }

  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  K.resize(n,n);

  // Compute cross covariance in place
  computeCrossCov(K, obsX, obsX, p);
}

// Compute covariance matrix K using default kernel paramaters
void GP::GaussianProcess::computeCov()
{
  computeCov(kernelParams);
}


// Compute Cholesky factorization of covariance matrix K with noise
void GP::GaussianProcess::computeChol(double noise)
{
  cholesky = ( K + (noise+jitter)*Matrix::Identity(static_cast<int>(K.cols()), static_cast<int>(K.cols())) ).llt();
}

// Compute Cholesky factorization of covariance matrix K using default noise
void GP::GaussianProcess::computeChol()
{
  computeChol(noiseLevel);
}


// Evaluate NLML for specified kernel hyperparameters p
// (note: alpha is used to share calculations with DNLML function)
double GP::GaussianProcess::evalNLML(const Vector & p, Matrix & alpha)
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  
  auto pcopy = static_cast<Vector>(p);
  if (!fixedNoise)
    {
      auto noise = static_cast<double>((pcopy.head(1))[0]);
      auto params = static_cast<Vector>( pcopy.tail(paramCount) );
      computeCov(params);
      computeChol(noise);
    }
  else
    {
      auto params = static_cast<Vector>( pcopy );
      computeCov(params);
      computeChol();
    }


  Matrix cholMat(cholesky.matrixL());
  alpha = cholesky.solve(obsY);

  auto NLML_value = -(-0.5 * (obsY.transpose()*alpha)(0)  -  cholMat.array().log().matrix().trace() - 0.5*n*std::log(2*PI));
  //std::cout << NLML_value << std::endl;
  return NLML_value;
}


// Evaluate derivatives of NLML for specified kernel hyperparameters
// (note: alpha is used to retrieve calculations from NLML function)
void GP::GaussianProcess::evalDNLML(const Vector & p, Vector & g, Matrix & alpha)
{
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  auto pcopy = static_cast<Vector>(p);

  Vector params;
  if (!fixedNoise)
    params = static_cast<Vector>( pcopy.tail(paramCount) );
  else
    params = static_cast<Vector>(pcopy);

  // Specify noise derivative if using trainable noise
  // and adjust array indices using shift accordingly
  int shift = 0;
  if (!fixedNoise)
      g[0] = -0.5 * ( alpha*alpha.transpose() - cholesky.solve(Matrix::Identity(n,n)) ).trace() ;
  else
    shift = 1;

  Matrix dK_i(n,n);
  for (auto i : boost::irange(1,paramCount+1))
    {
      computeCrossCov(dK_i, obsX, obsX, params, i);
      g[i-shift] = -0.5 * ( alpha*alpha.transpose()*dK_i - cholesky.solve(dK_i) ).trace() ;
    }
}


// Fit model hyperparameters
void GP::GaussianProcess::fitModel()
{
  // Get combined parameter/noise vector size 
  int n = (fixedNoise) ? static_cast<int>(kernelParams.size()) : static_cast<int>(kernelParams.size()) + 1 ;
  
  //fminProblem problem( std::make_unique<GaussianProcess>(*this) );
  auto lb = static_cast<Vector>((Eigen::VectorXd::Ones(n) * 0.001));
  auto ub = static_cast<Vector>((Eigen::VectorXd::Ones(n) * 500.0));

  // Adjust bounds for noise level parameter
  if (!fixedNoise)
    {
      lb[0] = 0.00001;
      ub[0] = 10.0;
    }

  //fminProblem problem( std::make_unique<GaussianProcess>(*this) , lb, ub );
  fminProblem problem( this , lb, ub );
  problem.setLowerBound(lb);
  problem.setUpperBound(ub);

  
  // Initialize L-BFGS solver
  //cppoptlib::BfgsSolver<fminProblem> solver;
  cppoptlib::LbfgsbSolver<fminProblem> solver;

  // Specify stopping criteria
  cppoptlib::Criteria<double> crit = cppoptlib::Criteria<double>::defaults();
  crit.iterations = 5000;
  crit.gradNorm = 10.0; //!< Minimum norm of gradient vector
  //crit.xDelta = 0;      //!< Minimum change in parameter vector
  //crit.fDelta = 0;      //!< Minimum change in cost function
  //crit.condition = 0;
  solver.setStopCriteria(crit);

  // Specify initial parameter values for solver
  Vector x = static_cast<Vector>(Eigen::VectorXd::Ones(n));
  if (!fixedNoise)
    x[0] = 0.1;

  // Use solver to find argmin of NLML corresponding to the optimal parameters
  solver.minimize(problem, x);
  //std::cout << "argmin      " << x.transpose() << std::endl;
  //std::cout << "f in argmin " << problem(x) << std::endl;

  // Assign tuned parameters to model
  if (!fixedNoise)
    {
      noiseLevel = static_cast<double>((x.head(1))[0]);
      kernelParams = x.tail(x.size() - 1);
    }
  else
    kernelParams = x;

  // Recompute covariance and Cholesky factor
  computeCov();
  computeChol();
};


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


// Show covariance matrix K
void GP::GaussianProcess::showCov(int prec)
{
  // Check if observations have been defined
  if ( N == 0 ) { std::cout << "[*] No observations (use 'setObs' to define observations).\n"; return; }

  using format = std::ios_base::fmtflags;
  using precision = std::streamsize;
  format initialState = std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  precision initialPrecision = std::cout.precision(prec);

  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());
  for (auto i : boost::irange(0,n) )
    {
      for (auto j : boost::irange(0,n) )
        {
          std::cout.width(prec+3);
          std::cout << K(i,j) << ((j<n-1) ? " " : "\n");
        }
    }

  // Restore formatting settings
  std::cout.setf(initialState, std::ios_base::floatfield);
  std::cout.precision(initialPrecision);
}


// Show observations
void GP::GaussianProcess::showObs()
{
  // Check if observations have been defined
  if ( N == 0 ) { std::cout << "[*] No observations (use 'setObs' to define observations).\n"; return; }
  
  using format = std::ios_base::fmtflags;
  using precision = std::streamsize;

  format initialState = std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  precision initialPrecision = std::cout.precision(2);
  int width = 7;

  std::cout << std::endl << "Input Observations:" << std::endl;
  
  for ( auto i : boost::irange(0, static_cast<int>(obsX.rows()) ) )
    {
      std::cout.width(width);
      std::cout << obsX(i) << "  -> ";
      std::cout.width(width);
      std::cout << obsY(i) << std::endl;
    }

  std::cout.setf(initialState, std::ios_base::floatfield);
  std::cout.precision(initialPrecision);
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
