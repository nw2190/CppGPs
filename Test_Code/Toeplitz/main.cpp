// main.cpp -- test GP class definitions
//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <cmath>
#include <string>
#include <array>
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

// Define function for retrieving time from chrono
float getTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
  return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);
};

// Specify the true target function
double targetFunc(double x)
{
  return std::sin(3.0*(x-0.1))*(0.5-(x-0.1))*15.0;
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
  using time = high_resolution_clock::time_point;
  
  // Set random seed
  //std::srand(static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));
  std::srand(static_cast<unsigned int>(0));
  //std::srand(static_cast<unsigned int>(std::time(0)));

  // Initialize Gaussian process model
  GaussianProcess model;

  // Specify observation data
  //int obsCount = 500;
  int obsCount = 250;
  //int obsCount = 100;
  //int obsCount = 20;
  Matrix x = sampleUnif(0.0, 1.0, obsCount);
  Matrix y;
  //auto noiseLevel = 0.05;
  auto noiseLevel = 0.025;
  auto noise = Eigen::VectorXd::Random(obsCount) * noiseLevel;
  //x.resize(obsCount, 1);
  //x = linspace(0.0, 1.0, obsCount);
  y.resize(obsCount, 1);
  y = x.unaryExpr(std::ptr_fun(targetFunc)) + noise;
  model.setObs(x,y);

  // Fix noise level
  //model.setNoise(std::pow(noiseLevel, 2));
  //model.setNoise(noiseLevel);
  //model.setNoise(std::sqrt(noiseLevel));
  model.setNoise(0.000805);
  
  // Define initial kernel parameters
  //Vector params(2);
  //params << 1.0, 1.0;
  //Vector params(1);
  //params << 1.0;

  RBF kernel;
  //model.setKernel(kernel);
  model.setKernel(kernel);
  //model.setKernel(std::make_unique<GP::Kernel>(kernel));
  //model.setKernel(std::make_shared<GP::Kernel>(kernel));


  // Define kernel for GP model
  //model.setKernel( std::make_unique<GP::kernelfn>(kernel) , params );
  //model.setDistKernel( std::make_unique<GP::distkernelfn>(distKernel) , params );

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

  // Get sample paths from posterior
  int sampleCount = 100;
  Matrix samples = model.getSamples(sampleCount);

  
  // Save true and predicted means/variances to file
  std::string outputFile = "predictions.csv";
  Matrix trueSoln = testMesh.unaryExpr(std::ptr_fun(targetFunc));
  std::ofstream fout;
  fout.open(outputFile);
  for ( auto i : boost::irange(0,predCount) )
    {
      fout << testMesh(i) << "," << trueSoln(i) << "," << pmean(i) << "," << pvar(i) << "\n";
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
