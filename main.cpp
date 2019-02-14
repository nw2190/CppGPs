// main.cpp -- example use of CppGPs for Gaussian process regression

// Eigen macros for using externel routines (may need to be moved to header file)
//#define EIGEN_USE_BLAS            // Must also link to BLAS library
//#define EIGEN_USE_LAPACKE_STRICT  // Must also link to LAPACKE library
//#define EIGEN_USE_MKL_ALL         // Must also link to Intel MKL library
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <chrono>
#include <fstream>
#include <boost/range/irange.hpp>
#include "GPs.h"


// Specify the target function for Gaussian process regression
double targetFunc(double x)
{
  double oscillation = 30.0;
  //double oscillation = 20.0;
  //double oscillation = 10.0;
  return std::sin(oscillation*(x-0.1))*(0.5-(x-0.1))*15.0;
}


// Example use of CppGPs code for Gaussian process regression
int main(int argc, char const *argv[])
{

  // Retrieve aliases from GP namescope
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  // Convenience using-declarations
  using std::cout;
  using std::endl;
  using GP::GaussianProcess;
  using GP::linspace;
  using GP::sampleUnif;
  using GP::RBF;
  
  // Used for timing code with chrono
  using GP::high_resolution_clock;
  using GP::time;
  using GP::getTime;
  
  // Set random seed based on system clock
  std::srand(static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));

  // Fix random seed for debugging and testing
  //std::srand(static_cast<unsigned int>(0));


  //
  //   [ Define Training Data ]
  //

  // Specify observation data count
  int obsCount = 1000;

  // Specify observation noise level
  auto noiseLevel = 1.0;

  // Define random uniform noise to add to target observations
  auto noise = Eigen::VectorXd::Random(obsCount) * noiseLevel;

  // Define observations by sampling random uniform distribution
  Matrix X = sampleUnif(0.0, 1.0, obsCount);
  Matrix y;  y.resize(obsCount, 1);

  // Define target observations 'y' by applying 'targetFunc'
  // to the input observations 'X' and adding a noise vector
  y = X.unaryExpr(std::ptr_fun(targetFunc)) + noise;


  
  //
  //   [ Construct Gaussian Process Model ]
  //
  
  // Initialize Gaussian process model
  GaussianProcess model;
  
  // Specify training observations for GP model
  model.setObs(X,y);

  // Fix noise level [ noise level is fit to the training data by default ]
  //model.setNoise(0.33);
  
  // Initialize a RBF covariance kernel and assign it to the model
  RBF kernel;
  model.setKernel(kernel);

  // Specify hyperparameter bounds  [ including noise bounds (index 0) ]
  // Vector lbs(2); lbs << 0.0001 , 0.01;
  // Vector ubs(2); ubs << 5.0    , 100.0;
  // model.setBounds(lbs, ubs);

  // Specify hyperparameter bounds  [ excluding noise bounds ]
  Vector lbs(1);  lbs <<  0.01;
  Vector ubs(1);  ubs <<  100.0;
  model.setBounds(lbs, ubs);
  

  // Fit covariance kernel hyperparameters to the training data
  time start = high_resolution_clock::now();
  model.fitModel();  
  time end = high_resolution_clock::now();
  auto computationTime = getTime(start, end);

  // Display computation time required for fitting the GP model
  cout << "\nComputation Time: ";
  cout << computationTime << " s" << endl;

  // Retrieve the tuned/optimized kernel hyperparameters
  auto optParams = model.getParams();
  cout << "\nOptimized Hyperparameters:" << endl << optParams.transpose() << "  ";
  auto noiseL = model.getNoise();
  cout << "(Noise = " << noiseL << ")\n" << endl;

  // Display the negative log marginal likelihood (NLML) of the optimized model
  cout << "NLML:  " << model.computeNLML() << endl << endl;


  //
  //   [ Posterior Predictions and Samples ]
  //

  // Define test mesh for GP model predictions
  int predCount = 100;
  auto testMesh = linspace(0.0, 1.0, predCount);
  model.setPred(testMesh);

  // Compute predicted means and variances for the test points
  model.predict();
  Matrix pmean = model.getPredMean();
  Matrix pvar = model.getPredVar();
  Matrix pstd = (pvar.array().sqrt()).matrix();

  // Get sample paths from the posterior distribution of the model
  int sampleCount = 100;
  Matrix samples = model.getSamples(sampleCount);



  //
  //   [ Save Results for Plotting ]
  //
  
  // Save true and predicted means/variances to file
  std::string outputFile = "predictions.csv";
  Matrix trueSoln = testMesh.unaryExpr(std::ptr_fun(targetFunc));
  std::ofstream fout;
  fout.open(outputFile);
  for ( auto i : boost::irange(0,predCount) )
      fout << testMesh(i) << "," << trueSoln(i) << "," << pmean(i) << "," << pstd(i) << "\n";
  fout.close();

  // Save observations to file
  std::string outputObsFile = "observations.csv";
  fout.open(outputObsFile);
  for ( auto i : boost::irange(0,obsCount) )
      fout << X(i) << "," << y(i) << "\n";
  fout.close();
  
  // Save samples to file
  std::string outputSampleFile = "samples.csv";
  fout.open(outputSampleFile);
  for ( auto j : boost::irange(0,sampleCount) )
    {
      for ( auto i : boost::irange(0,predCount) )
          fout << samples(i,j) << ((i<predCount-1) ? "," : "\n");
    }
  fout.close();

  
  return 0;
}
