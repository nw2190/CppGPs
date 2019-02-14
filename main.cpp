// main.cpp -- test GP class definitions
//#define EIGEN_USE_MKL_ALL
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

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Declare undefined class for checking deduced types
template<typename T>
class CheckType;

// Specify the true target function
double targetFunc(double x)
{
  double oscillation = 30.0;
  //double oscillation = 20.0;
  //double oscillation = 10.0;
  return std::sin(oscillation*(x-0.1))*(0.5-(x-0.1))*15.0;
}

// Example using GP class for regression
int main(int argc, char const *argv[])
{
  using std::cout;
  using std::endl;
  using GP::GaussianProcess;
  using GP::linspace;
  using GP::sampleUnif;
  using GP::RBF;
  
  // Aliases for timing functions with chrono
  using std::chrono::high_resolution_clock;
  //using time = high_resolution_clock::time_point;
  using time = GP::time;
  using GP::getTime;
  
  // Set random seed based on system clock
  std::srand(static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));

  // Fix random seed for debugging and testing
  //std::srand(static_cast<unsigned int>(0));

  // Initialize Gaussian process model
  GaussianProcess model;

  // Specify observation data count
  //int obsCount = 2000;
  int obsCount = 1000;
  //int obsCount = 500;
  //int obsCount = 250;  
  //int obsCount = 10;

  // Specify observation noise level
  //auto noiseLevel = 0.05;
  //auto noiseLevel = 0.15;
  //auto noiseLevel = 0.25;
  auto noiseLevel = 1.0;
  auto noise = Eigen::VectorXd::Random(obsCount) * noiseLevel;

  // Define observations
  Matrix x = sampleUnif(0.0, 1.0, obsCount);
  //x.resize(obsCount, 1);
  //x = linspace(0.0, 1.0, obsCount);
  Matrix y;
  y.resize(obsCount, 1);
  y = x.unaryExpr(std::ptr_fun(targetFunc)) + noise;
  model.setObs(x,y);

  // Fix noise level
  //model.setNoise(0.00019);
  
  // Specify covariance kernel
  RBF kernel;
  model.setKernel(kernel);

  // Define hyperparameter bounds
  /*
  Vector lbs(2);
  lbs << 0.0001 , 0.01;
  Vector ubs(2);
  ubs << 5.0 , 5.0;
  model.setBounds(lbs, ubs);
  */
  
  Vector lbs(1);
  lbs <<  0.01;
  Vector ubs(1);
  ubs <<  100.0;
  model.setBounds(lbs, ubs);
  

  // Fit kernel hyperparameters to data
  time start = high_resolution_clock::now();
  model.fitModel();  
  time end = high_resolution_clock::now();
  auto computationTime = getTime(start, end);

  // Display computation time
  cout << "\nComputation Time: ";
  cout << computationTime << " s" << endl;

  // Get tuned hyperparameters
  auto optParams = model.getParams();
  cout << "\nOptimized Hyperparameters:" << endl << optParams.transpose() << "  ";
  auto noiseL = model.getNoise();
  cout << "(Noise = " << noiseL << ")\n" << endl;


  // Define test mesh for predictions
  int predCount = 100;
  auto testMesh = linspace(0.0, 1.0, predCount);
  model.setPred(testMesh);

  // Compute predicted means and variances
  model.predict();
  Matrix pmean = model.getPredMean();
  Matrix pvar = model.getPredVar();
  Matrix pstd = (pvar.array().sqrt()).matrix();

  // Get sample paths from posterior
  int sampleCount = 100;
  Matrix samples = model.getSamples(sampleCount);

  // Compare NLML results
  cout << "NLML:  " << model.computeNLML() << endl << endl;


  
  // Save true and predicted means/variances to file
  std::string outputFile = "predictions.csv";
  Matrix trueSoln = testMesh.unaryExpr(std::ptr_fun(targetFunc));
  std::ofstream fout;
  fout.open(outputFile);
  for ( auto i : boost::irange(0,predCount) )
    {
      fout << testMesh(i) << "," << trueSoln(i) << "," << pmean(i) << "," << pstd(i) << "\n";
    }
  fout.close();

  // Save observations to file
  std::string outputObsFile = "observations.csv";
  fout.open(outputObsFile);
  for ( auto i : boost::irange(0,obsCount) )
    {
      fout << x(i) << "," << y(i) << "\n";
    }
  fout.close();
  
  // Save samples to file
  std::string outputSampleFile = "samples.csv";
  fout.open(outputSampleFile);
  for ( auto j : boost::irange(0,sampleCount) )
    {
      for ( auto i : boost::irange(0,predCount) )
        {
          fout << samples(i,j) << ((i<predCount-1) ? "," : "\n");
        }
    }
  fout.close();

  
  return 0;
}
