#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <memory>
#include <thread>
#include <random>
#include <limits>
#include <boost/range/irange.hpp>
#include <Eigen/Dense>
#include "GPs.h"

// Retrieve aliases from GP namescope
using Matrix = GP::Matrix;
using Vector = GP::Vector;


// Compute pairwise distance between lists of points
void GP::pdist(Matrix & Dv, Matrix & X1, Matrix & X2)
{
  auto n = static_cast<int>(X1.rows());
  auto entryCount = static_cast<int>( (n*(n-1))/2);
  Dv.resize(entryCount, 1);

  // Get thread count
  int threadCount = Eigen::nbThreads( );
      
  // Get problem dimension d per thread
  auto d = static_cast<int>(n/threadCount);
  
  std::vector<int> startVals;
  for ( auto i : boost::irange(0,threadCount) )
    startVals.emplace_back(i*d);

  std::vector<int> endVals;
  for ( auto i : boost::irange(1,threadCount) )
    endVals.emplace_back(i*d);
  endVals.emplace_back(n-1);

  // Define lambda function specifying each threads solver task
  auto lambda = [&Dv,&X1,&X2,n](int startInd, int endInd) {
                  for ( auto i : boost::irange(startInd, endInd) )
                    {      
                      for ( auto j : boost::irange(i+1,n) )
                        Dv(static_cast<int>(i*n-(i*(i+1))/2+j-i-1), 0) = (static_cast<Vector>(X1.row(i)-X2.row(j))).squaredNorm();
                    }
                };

  // Initialize thread list
  std::vector<std::thread> threadList;

  // Assign tasks to threads
  for ( auto i : boost::irange(0,threadCount) )
    threadList.emplace_back(lambda,startVals[i],endVals[i]);

  // Join threads
  for ( auto & thread : threadList )
    thread.join();

  
}

// Re-assemble pairwise distances into a dense matrix
void GP::squareForm(Matrix & D, Matrix & Dv, int n, double diagVal)
{
  D.resize(n,n);

  // Get thread count
  int threadCount = Eigen::nbThreads( );
      
  // Get problem dimension d per thread
  auto d = static_cast<int>(n/threadCount);
  
  std::vector<int> startVals;
  for ( auto i : boost::irange(0,threadCount) )
    startVals.emplace_back(i*d);

  std::vector<int> endVals;
  for ( auto i : boost::irange(1,threadCount) )
    endVals.emplace_back(i*d);
  endVals.emplace_back(n-1);

  // Define lambda function specifying each threads solver task
  auto lambda = [&D,&Dv,n](int startInd, int endInd) {
                  for ( auto i : boost::irange(startInd,endInd) )
                    {
                      for ( auto j : boost::irange(i+1, n) )
                       D(i,j) = D(j,i) = Dv(static_cast<int>(i*n-(i*(i+1))/2+j-i-1), 0);
                    }
                };

  // Initialize thread list
  std::vector<std::thread> threadList;

  // Assign tasks to threads
  for ( auto i : boost::irange(0,threadCount) )
    threadList.emplace_back(lambda,startVals[i],endVals[i]);

  // Join threads
  for ( auto & thread : threadList )
    thread.join();

  
  // Add diagonal values to distance matrix
  D.diagonal() = diagVal * Eigen::MatrixXd::Ones(n,1);
}


// Parse kernel parameter vector, separating noise from the kernel hyperparameters
void GP::Kernel::parseParams(const Vector & params, Vector & kernelParams, std::vector<double> & nonKernelParams)
{
  if ( !fixedNoise )
    {
      // Noise = params(0)
      nonKernelParams.emplace_back(params(0));
      if ( !fixedScaling )
        nonKernelParams.emplace_back(params(1));
      else
        nonKernelParams.emplace_back(scalingLevel);
    }
  else
    {
      // Scaling = params(0)
      nonKernelParams.emplace_back(noiseLevel);
      if ( !fixedScaling )
        nonKernelParams.emplace_back(params(0));
      else
        nonKernelParams.emplace_back(scalingLevel);
    }

  // Trim kernel parameter vector
  kernelParams = params.tail(paramCount);

}

// Compute covariance matrix (and gradients) from a vector of squared pairwise distances Dv
void GP::RBF::computeCov(Matrix & K, Matrix & obsX, Vector & params, std::vector<Matrix> & gradList, double jitter, bool evalGrad)
{
  auto n = static_cast<int>(K.rows());

  // Separate noise and scaling parameters from kernel hyperparameters
  Vector kernelParams;
  std::vector<double> noiseAndScaling;
  parseParams(params, kernelParams, noiseAndScaling);
  double noise = noiseAndScaling[0];
  double scaling = noiseAndScaling[1];
  
  // Compute distance matrix for each call
  Matrix Dv;
  pdist(Dv, obsX, obsX);

  // Evaluate covariance kernel on pairwise distance vector
  Matrix Kv;
  Kv.noalias() = scaling * ( (-0.5 / std::pow(kernelParams(0),2)) * Dv ).array().exp().matrix();

  // Make sure not to scale the jitter and noise terms
  squareForm(K, Kv, n, scaling*1.0 + jitter + noise);

  // Compute gradients w.r.t. kernel hyperparameters
  if ( evalGrad )
    {
      // Prepend gradient list with scaling term
      //int index = 0;
      //if (!fixedScaling)
      //  {
      //    Matrix dK_scaling;
      //    squareForm(dK_scaling, Kv, n, -noise);
      //    gradList[index++] = dK_scaling;
      //  }
      
      Matrix dK_i;
      Matrix dK_iv;
      dK_iv.noalias() = 1/std::pow(kernelParams(0),2) * ( Dv.array() * Kv.array() ).matrix();
      squareForm(dK_i, dK_iv, n); // Note: diagVal = 0.0
      gradList[0] = dK_i.eval();
      //gradList[index] = dK_i.eval();
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


// Evaluate NLML for specified kernel hyperparameters p
double GP::GaussianProcess::evalNLML(const Vector & p, Vector & g, bool evalGrad)
{
  time EVAL_start = high_resolution_clock::now();
  
  // Get matrix input observation count
  auto n = static_cast<int>(obsX.rows());

  // ASSUME OPTIMIZATION OVER LOG VALUES
  auto params = static_cast<Vector>(p);
  params = params.array().exp().matrix();

  // Compute covariance matrix and store Cholesky factor
  Matrix K(n,n);
  time start = high_resolution_clock::now();
  (*kernel).computeCov(K, obsX, params, gradList, jitter, evalGrad);
  time end = high_resolution_clock::now();
  time_computecov += getTime(start, end);


  start = high_resolution_clock::now();
  Eigen::LLT<Matrix> _cholesky(n);
  _cholesky = K.llt();
  end = high_resolution_clock::now();
  time_cholesky_llt += getTime(start, end);

  start = high_resolution_clock::now();
  Matrix _alpha = _cholesky.solve(obsY);
  end = high_resolution_clock::now();
  time_alpha += getTime(start, end);
  
  // Compute NLML value
  start = high_resolution_clock::now();
  double NLML_value = (obsY.transpose()*_alpha)(0);
  NLML_value += n*std::log(2*PI);
  NLML_value *= 0.5;
  NLML_value += _cholesky.matrixLLT().diagonal().array().log().sum();
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
      
      // Direct Implementation
      //Matrix term(n,n);
      //term.noalias() = _cholesky.solve(Matrix::Identity(n,n));
      //term.noalias() -= _alpha*_alpha.transpose();

      // Using Solve In Place
      //Matrix term = Matrix::Identity(n,n);
      //_cholesky.solveInPlace(term);
      //term.noalias() -= _alpha*_alpha.transpose();

      
      //
      //  MULTI-THREADED IMPLEMENTATION
      //
      Matrix term = Matrix::Identity(n,n);

      // Get thread count
      int threadCount = Eigen::nbThreads( );
      
      // Get problem dimension d per thread
      auto d = static_cast<int>(n/threadCount);

      // Construct partitioned list of block terms for solver
      std::vector<Matrix> termList;      
      for ( auto i : boost::irange(0,threadCount-1) )
        termList.emplace_back(term.block(0,i*d,n,d));

      // Ensure the final block extends to column n (i.e. account for index roundoff)
      termList.emplace_back(term.block(0, (threadCount-1)*d, n, n - (threadCount-1)*d) );

      // Define lambda function specifying each threads solver task
      auto lambda = [&termList,&_cholesky](int i) { _cholesky.solveInPlace(termList[i]); };

      // Initialize thread list
      std::vector<std::thread> threadList;

      // Assign tasks to threads
      for ( auto i : boost::irange(0,threadCount) )
        threadList.emplace_back(lambda,i);

      // Join threads
      for ( auto & thread : threadList )
        thread.join();

      // Reassemble blocks from solver threads back into original matrix
      for ( auto i : boost::irange(0,threadCount-1) )
        term.block(0,i*d,n,d) = termList[i];
      term.block(0, (threadCount-1)*d, n, n - (threadCount-1)*d) = termList[threadCount-1];
      
      // Compute final multiplicative term:  K^-1 - alpha*alpha^T
      term.noalias() -= _alpha*_alpha.transpose();

      
      end = high_resolution_clock::now();
      time_term += getTime(start, end);


      start = high_resolution_clock::now();      
      // Compute gradient for noise term if 'fixedNoise=false'
      int index = 0;
      double noise;
      if (!fixedNoise)
        {
          // Specify gradient of white noise kernel  [ dK_i = params(0)*Matrix::Identity(n,n) ]
          //g(index++) = 0.5 * (term * params(0)).trace() ;
          noise = params(0);
          g(index) = term.trace();
          g(index++) *= 0.5 * noise;
          //g(index++) *= 0.5 * params(0);
        }
      else
        noise = noiseLevel;

      
      if (!fixedScaling)
        {
          if (!fixedNoise)
            {
              double trace = 0.0;
              for ( auto j : boost::irange(0, n) )
                trace -= _alpha(j)*obsY(j);
              trace += n;
              trace *= 0.5;
              // Add  1/2 * trace[(noise * term)]  (i.e. the noise gradient)
              trace -= g(index-1);
              g(index++) = trace;
            }
          else
            {
              double trace = 0.0;
              Matrix dK = K - noise*Matrix::Identity(n,n);
              for ( auto j : boost::irange(0, n) )
                trace += term.row(j)*dK.col(j);
              g(index++) = 0.5*trace;
            }
          //  
          //  NOTE: The following implementation does not account for noise term (!)
          //
          //  Ah, it's so close though...
          //
          //  The adjusted covariance matrix is:  K'  =  s * K  +  noise * I  
          //
          //  But if we had   K' = s*K   and set   t = log(s) ~ s = exp(t) :
          //
          //  ( so that  dK'/dt  =  d/dt[ s*K ]  =  d/ds[ s*K ] * ds/dt  =  s * K  =  K'  )
          //
          //  then it would give us the reduction:
          //
          //  d/dt -log p(y|X,t)  =  -1/2 * trace[ ( (K'^-1 y)(K'^-1 y)^T - K'^-1 ) dK'/dt  ]
          //
          //   =  -1/2 * trace[ ( K'^-1 y y^T K'^-1 - K'^-1 ) dK'/dt  ]
          //
          //   =  -1/2 * trace[ ( K'^-1 y y^T K'^1 - K'^-1 ) K'  ]
          //
          //   =  -1/2 * trace[  (K'^-1 y) y^T - I  ]
          //
          //   =  -1/2 * trace[  alpha * y^T - I  ]
          //
          //
          //  but...    we do have:
          //  
          //   dK'/dt  =  d/dt[s*K]  =  s * K  =  K' - noise * I
          //
          //  so that the (corected) calculation above still yields:
          //
          //   =  -1/2 * trace[  (alpha * y^T - I)  -  noise * term  ]
          //
          //  and the trace of "term" has already been calculated...
          //
          //
          //double trace = 0.0;
          //for ( auto i : boost::irange(0,n) )
          //  trace -= _alpha(i)*obsY(i);
          //trace += n;
          //g(index++) = 0.5 * trace;
        }

      // Specify gradient w.r.t. the kernel scaling parameter
      //if (!fixedScaling)
      //  gradList.insert(gradList.begin(), K - noise*Matrix::Identity(n,n));
      
      // Compute gradients with respect to kernel hyperparameters
      for (auto dK_i = gradList.begin(); dK_i != gradList.end(); ++dK_i) 
        {
          // Compute trace of full matrix
          //g(index++) = 0.5 * (term * (*dK_i) ).trace() ;

          // POSSIBLE MULTI-THREADED IMPLEMENTATION
          // Construct zero initialized vector of partial trace values
          Matrix traceVals = Eigen::MatrixXd::Zero(threadCount,1);

          std::vector<int> startVals;
          for ( auto i : boost::irange(0,threadCount) )
            startVals.emplace_back(i*d);

          std::vector<int> endVals;
          for ( auto i : boost::irange(1,threadCount) )
            endVals.emplace_back(i*d);
          endVals.emplace_back(n);

          // Define lambda function specifying each threads solver task
          auto lambda = [&term,&dK_i,&traceVals](int i, int startInd, int endInd) {
                          for ( auto j : boost::irange(startInd, endInd) )
                            traceVals(i) += term.row(j)*(*dK_i).col(j);
                        };

          // Initialize thread list
          std::vector<std::thread> traceThreadList;

          // Assign tasks to threads
          for ( auto i : boost::irange(0,threadCount) )
            traceThreadList.emplace_back(lambda,i,startVals[i],endVals[i]);

          // Join threads
          for ( auto & thread : traceThreadList )
            thread.join();

          // Compute final trace value for derivative calculation
          g(index++) = 0.5*(traceVals.sum());
          
        }
      end = high_resolution_clock::now();
      time_grad += getTime(start, end);

      // Update gradient evaluation count
      gradientEvals += 1;

    }
  
  time EVAL_end = high_resolution_clock::now();
  time_evaluation += getTime(EVAL_start, EVAL_end);
  return NLML_value;
  
}


// Define simplified interface for evaluating NLML without gradient calculation
double GP::GaussianProcess::evalNLML(const Vector & p)
{
  Vector nullGrad(0);
  return evalNLML(p,nullGrad,false);
}


int GP::GaussianProcess::getAugParamCount(int count)
{
  if (!fixedNoise)
    {
      if (!fixedScaling)
        return count + 2;
      else
        return count + 1;
    }
  else
    {
      if (!fixedScaling)
        return count + 1;
      else
        return count;
    }
}

// Fit model hyperparameters
void GP::GaussianProcess::fitModel()
{

  // Get combined parameter/noise vector size
  paramCount = (*kernel).getParamCount();
  //augParamCount = (fixedNoise) ? static_cast<int>(paramCount) : static_cast<int>(paramCount) + 1 ;
  augParamCount = getAugParamCount(paramCount);

  // Pass noise level to kernel when 'fixedNoise=true'
  if ( fixedNoise )
    (*kernel).setNoise(noiseLevel);
  if ( fixedScaling )
    (*kernel).setScaling(scalingLevel);

  // Declare vector for storing gradient calculations
  Vector g(augParamCount);

  // Initialize gradient list with identity matrices
  //for ( auto i : boost::irange(0,augParamCount) )
  //int gradientCount = paramCount;
  //if (!fixedScaling)
  //  gradientCount += 1;
  //for ( auto i : boost::irange(0,gradientCount) )

  for ( auto i : boost::irange(0,paramCount) )
    {
      // Avoid compiler warning for unused variable
      (void)i;
      gradList.push_back(Matrix::Identity(static_cast<int>(obsY.size()),static_cast<int>(obsY.size())));
    }


  // Convert hyperparameter bounds to log-scale
  Vector lbs, ubs;
  parseBounds(lbs, ubs, augParamCount);

  // Initialize optimal hyperparamter values
  Vector optParams = Eigen::MatrixXd::Zero(augParamCount,1);

  // Define restart count for optimizer
  int restartCount = solverRestarts;

  // Declare variables to store optimization loop results
  double currentVal;
  double optVal = 1e9;
  Vector theta(augParamCount);

  // Define low-precision solver for restart loop
  LBFGSpp::LBFGSParam<double> param;
  param.linesearch  = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
  param.m = 10;
  param.epsilon = 1e-6;
  param.max_iterations = 10;
  param.max_linesearch = 5;
  param.delta = 1e-4;

  int niter;
  
  // Evaluate optimizer with various different initializations
  for ( auto i : boost::irange(0,restartCount) )
    {
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

      // Create solver and function object
      LBFGSpp::LBFGSSolver<double> solver(param);
      niter = solver.minimize(*this, theta, currentVal);
      
      // Compute current NLML and store parameters if optimal
      if ( currentVal < optVal ) { optVal = currentVal; optParams = theta; }
      
    }

  // Perform one last optimization starting from best parameters so far
  //if ( restartCount == 0 ) 
  //  optParams = sampleUnifVector(lbs, ubs);
    

  LBFGSpp::LBFGSParam<double> finalparam;

  // Line Search Options
  //param.linesearch  = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
  //param.linesearch  = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
  //param.linesearch  = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

  // TRY MODELLING SCIPY fmin_l_bfgs_b PARAMETERS
  finalparam.m = 10;
  finalparam.epsilon = 1e-5;
  finalparam.max_linesearch = 20;
  double eps = 2.220446049250313e-16;
  double factr = solverPrecision;
  finalparam.past = 1;
  //finalparam.ftol = factr*eps;
  finalparam.delta = factr*eps;
  finalparam.max_iterations = 100;
  
  // Create solver and function object
  LBFGSpp::LBFGSSolver<double> finalsolver(finalparam);
  niter = finalsolver.minimize(*this, optParams, optVal);

  if ( VERBOSE )
    {
      std::cout << "\n[*] Solver Iterations = " << niter <<std::endl;
      std::cout << "\n[*] Function Evaluations = " << gradientEvals <<std::endl;
    }
  
  // ASSUME OPTIMIZATION OVER LOG VALUES
  optParams = optParams.array().exp().matrix();

  ///* [ This is included in the SciKit Learn model.fit() call as well ]

  // Recompute covariance and Cholesky factor
  auto n = static_cast<int>(obsX.rows());
  Matrix K(n,n);
  (*kernel).computeCov(K, obsX, optParams, gradList, jitter, false);
  cholesky = K.llt();
  alpha.noalias() = cholesky.solve(obsY);

  // Assign tuned parameters to model
  if (!fixedNoise)
    {
      noiseLevel = optParams[0];
      if (!fixedScaling)
          scalingLevel = optParams[1];
    }
  else
    {
      if (!fixedScaling)
        scalingLevel = optParams[0];
    }

  optParams = static_cast<Vector>(optParams.tail(paramCount));
  (*kernel).setParams(optParams);


  // DISPLAY TIMING INFORMATION
  if ( VERBOSE )
    {
      std::cout << "\n Time Diagnostics |\n";
      std::cout << "------------------\n";
      std::cout << "computeCov():\t  " << time_computecov/gradientEvals  << std::endl;
      std::cout << "cholesky.llt():\t  " << time_cholesky_llt/gradientEvals  << std::endl;
      std::cout << "_alpha term:\t  " << time_alpha/gradientEvals  << std::endl;
      std::cout << "NLML:\t  \t  " << time_NLML/gradientEvals  << std::endl;
      std::cout << "Grad term:\t  " << time_term/gradientEvals  << std::endl;
      std::cout << "Gradient:\t  " << time_grad/gradientEvals  << std::endl;
      std::cout << "\nEvaluation:\t  " << time_evaluation/gradientEvals  << std::endl;
    }
  
  
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
  kstar_and_v *= scalingLevel;
    
  // Compute covariance matrix for test points
  Matrix kstarmat;
  kstarmat.resize(m,m);
  (*kernel).computeCrossCov(kstarmat, predX, predX, params);
  kstarmat *= scalingLevel;

  // Set predictive means/variances and compute negative log marginal likelihood
  Matrix cholMat(cholesky.matrixL());
  //predMean.noalias() = kstar_and_v.transpose() * _alpha;
  predMean.noalias() = kstar_and_v.transpose() * alpha;
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
  //Matrix L = ( predCov + (noiseLevel+jitter)*Matrix::Identity(static_cast<int>(predCov.cols()), static_cast<int>(predCov.cols())) ).llt().matrixL();
  Matrix L = ( scalingLevel*predCov + (noiseLevel+jitter)*Matrix::Identity(static_cast<int>(predCov.cols()), static_cast<int>(predCov.cols())) ).llt().matrixL();

  // Draw samples using the formula:  y = m + L*u
  Matrix samples = predMean.replicate(1,count) + L*uVals;
  
  return samples;
}


// Evaluate NLML [public interface]
double GP::GaussianProcess::computeNLML(const Vector & p)
{
  // Compute log-hyperparameters
  Vector logparams(augParamCount);

  int index = 0;
  logparams(index++) = std::log(noiseLevel);
  logparams(index++) = std::log(scalingLevel);
  for ( auto i : boost::irange(index,augParamCount) )
    logparams(i) = std::log(p(i-index));

  // Evaluate NLML using log-hyperparameters
  return evalNLML(logparams);
}


// Evaluate NLML with default noise level [public interface]
double GP::GaussianProcess::computeNLML()
{
  auto params = (*kernel).getParams();
  return computeNLML(params);
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




// Define utility function for formatting hyperparameter bounds
void GP::GaussianProcess::parseBounds(Vector & lbs, Vector & ubs, int augParamCount)
{
  lbs.resize(augParamCount);
  ubs.resize(augParamCount);

  double defaultLowerBound = 0.01;
  double defaultUpperBound = 2.0;
  
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




/*
//  POSSIBLY UNNEEDED POINTWISE KERNEL DEFINITIONS

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

*/





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
//   SECOND MINIMIZATION IMPLEMENTATION USING CPPOPTLIB
//

// Fit model hyperparameters
//void GP::GaussianProcess::fitModel()

// Initialize gradient vector size
//cppOptLibgrad.resize(augParamCount);

/*
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

*/

// Display final solver criteria values
/*
std::cout << "\nSolver Criteria |";
std::cout << "\n----------------\n" << solver.criteria() << std::endl;
std::cout << "gradEvals =\t" << gradientEvals <<std::endl;
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
