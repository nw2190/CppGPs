// main.cpp -- example use of CppGPs for Gaussian process regression
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <chrono>
#include <fstream>
#include <boost/range/irange.hpp>
#include "GPs.h"

#include <limits>

// Specify the target function for Gaussian process regression
double targetFunc(Eigen::MatrixXd X)
{

  if ( X.size() == 1 )
    {
      // Define the target function to be an oscillatory, non-periodic function
      double oscillation = 30.0;
      double xshifted = 0.5*(X(0) + 1.0);
      return std::sin(oscillation*(xshifted-0.1))*(0.5-(xshifted-0.1))*15.0;
    }
  else
    {
      // Define the target function to be the Marr wavelet (a.k.a "Mexican Hat" wavelet)
      // ( see https://en.wikipedia.org/wiki/Mexican_hat_wavelet )
      double sigma = 0.25;
      double pi = std::atan(1)*4;
      double radialTerm = std::pow(X.squaredNorm()/sigma,2);
      return 2.0 / (std::sqrt(pi*sigma) * std::pow(pi,1.0/4.0)) * (1.0 - radialTerm) * std::exp(-radialTerm);
    }
}

// Specify multi-modal target function for testing
double targetFuncMultiModal(Eigen::MatrixXd X)
{
  Eigen::MatrixXd p1(1,2);  p1 << 0.25, 0.5;
  Eigen::MatrixXd p2(1,2);  p2 << -0.25, -0.5;
  double scale1 = 10.0; double scale2 = 15.0;
  double sigma1 = 0.4;  double sigma2 = 0.3;
  double radialTerm1 = ((X-p1)*(1/sigma1)).squaredNorm();
  double radialTerm2 = ((X-p2)*(1/sigma2)).squaredNorm();
  return scale1*std::exp(-radialTerm1) + scale2*std::exp(-radialTerm2);
}


// Example use of CppGPs code for Gaussian process regression
int main(int argc, char const *argv[])
{

  // Inform Eigen of possible multi-threading
  Eigen::initParallel();

  // Retrieve aliases from GP namescope
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  // Convenience using-declarations
  using std::cout;
  using std::endl;
  using GP::GaussianProcess;
  using GP::linspace;
  using GP::sampleNormal;
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

  // Specify the input dimensions
  //int inputDim = 1;
  int inputDim = 2;
  
  // Specify observation data count
  int obsCount;
  if ( inputDim == 1 )
    obsCount = 250;
  else
    obsCount = 2000;
    
  // Specify observation noise level
  auto noiseLevel = 1.0;

  // Define random noise to add to target observations
  //auto noise = sampleNormal(obsCount) * noiseLevel;
  Matrix noise;
  noise.noalias() = sampleNormal(obsCount) * noiseLevel;

  // Define observations by sampling random uniform distribution
  Matrix X = sampleUnif(-1.0, 1.0, obsCount, inputDim);
  Matrix y;  y.resize(obsCount, 1);
  
  // Define target observations 'y' by applying 'targetFunc'
  // to the input observations 'X' and adding a noise vector
  for ( auto i : boost::irange(0,obsCount) )
    y(i) = targetFunc(X.row(i)) + noise(i);
  //y(i) = targetFuncMultiModal(X.row(i)) + noise(i);


  
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

  // Specify solver precision
  if ( inputDim == 1 )
    model.setSolverPrecision(1e-4);
  else
    model.setSolverPrecision(1e-3);

  
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
  cout << "NLML:  " << std::fixed << std::setprecision(4) << model.computeNLML() << endl << endl;

  
  //
  //   [ Posterior Predictions and Samples ]
  //

  // Define test mesh for GP model predictions
  int predCount;
  if ( inputDim == 1 )
    predCount = 100;
  else
    predCount = 1000;
  int predRes = static_cast<int>(std::pow(predCount,1.0/inputDim));
  auto testMesh = linspace(-1.0, 1.0, predRes, inputDim);
  model.setPred(testMesh);

  // Reset predCount to account for rounding
  predCount = std::pow(predRes,inputDim);

  // Compute predicted means and variances for the test points
  model.predict();
  Matrix pmean = model.getPredMean();
  Matrix pvar = model.getPredVar();
  Matrix pstd = pvar.array().sqrt().matrix();

  // Get sample paths from the posterior distribution of the model
  int sampleCount;
  Matrix samples;
  if ( inputDim == 1 )
    {
      sampleCount = 25;
      samples = model.getSamples(sampleCount);
    }

  
  //
  //   [ Save Results for Plotting ]
  //
  
  // Save true and predicted means/variances to file
  std::string outputFile = "predictions.csv";
  Matrix trueSoln(predCount,1);
  for ( auto i : boost::irange(0,predCount) )
    trueSoln(i,0) = targetFunc(testMesh.row(i));
  //trueSoln(i,0) = targetFuncMultiModal(testMesh.row(i));

  std::ofstream fout;
  fout.open(outputFile);
  for ( auto i : boost::irange(0,predCount) )
    {
      for ( auto j : boost::irange(0,inputDim) )
        fout << testMesh(i,j) << ",";
      
      fout << trueSoln(i) << "," << pmean(i) << "," << pstd(i) << "\n";
    }
  fout.close();

  // Save observations to file
  std::string outputObsFile = "observations.csv";
  fout.open(outputObsFile);
  for ( auto i : boost::irange(0,obsCount) )
    {
      for ( auto j : boost::irange(0,inputDim) )
        fout << X(i,j) << ",";
      
      fout << y(i) << "\n";
    }
  fout.close();


  if ( inputDim == 1 )
    {
      // Save samples to file
      std::string outputSampleFile = "samples.csv";
      fout.open(outputSampleFile);
      for ( auto j : boost::irange(0,sampleCount) )
        {
          for ( auto i : boost::irange(0,predCount) )
            fout << samples(i,j) << ((i<predCount-1) ? "," : "\n");
        }
      fout.close();
    }

  
  return 0;
  
}
