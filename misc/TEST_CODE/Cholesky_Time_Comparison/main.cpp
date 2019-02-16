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
  int obsCount = 2000;
    
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

  // Specify solver precision and iteration count
  model.setSolverPrecision(1e-3);
  model.setSolverIterations(1);
    
  // Fit covariance kernel hyperparameters to the training data
  model.fitModel();  


  Matrix K = model.getK();
  Matrix alpha = model.getAlpha();
  time start = high_resolution_clock::now();
  auto cholesky = K.llt();
  time end = high_resolution_clock::now();
  auto choleskyTime = getTime(start, end);

  
  start = high_resolution_clock::now();
  // Try forcing Eigen to solve in place
  Matrix term;
  term.noalias() = Matrix::Identity(obsCount,obsCount);
  cholesky.solveInPlace(term);
  term.noalias() -= alpha*alpha.transpose();
  end = high_resolution_clock::now();
  auto computationTime = getTime(start, end);


  cout << "\n Cholesky Time:\t\t" << choleskyTime << endl;
  cout << "\n Computation Time:\t" << computationTime << endl << endl;
  

  //
  //   [ Save Results for Comparison ]
  //
  
  // Save K matrix
  std::string outputFile = "K.csv";
  std::ofstream fout;
  fout << std::scientific << std::setprecision(64);
  fout.open(outputFile);
  for ( auto i : boost::irange(0,obsCount) )
    {
      for ( auto j : boost::irange(0,obsCount) )
        fout << K(i,j) << ((j<obsCount-1) ? "," : "\n");
    }
  fout.close();

  // Save term matrix
  outputFile = "term.csv";
  fout.open(outputFile);
  for ( auto i : boost::irange(0,obsCount) )
    {
      for ( auto j : boost::irange(0,obsCount) )
        fout << term(i,j) << ((j<obsCount-1) ? "," : "\n");
    }
  fout.close();


  // Save alpha matrix
  outputFile = "alpha.csv";
  fout.open(outputFile);
  for ( auto i : boost::irange(0,obsCount) )
    fout << alpha(i) << ((i<obsCount-1) ? "," : "\n");
  fout.close();
  
  
  return 0;
  
}
