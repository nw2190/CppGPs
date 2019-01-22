// main.cpp -- test GP class definitions
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
/*
// Define squared exponential kernel between two points
double kernel(Matrix & x, Matrix & y, Vector & params, int n)
{
  switch (n)
    {
    case 0: return params(0) * std::exp( -(x-y).squaredNorm() / (2.0*std::pow(params(1),2)));
    case 1: return std::exp( -(x-y).squaredNorm() / (2.0*std::pow(params(1),2)));
    case 2: return params(0) * (x-y).squaredNorm() / std::pow(params(1),3) * std::exp( -(x-y).squaredNorm() / (2.0*std::pow(params(1),2)));
    default: std::cout << "\n[*] UNDEFINED DERIVATIVE\n"; return 0.0;
    }
}

// Define squared exponential kernel provided a squared distance as input
//double distKernel(double d, Vector & params)
//{ return params(0) * std::exp( -d / (2.0*std::pow(params(1),2))); }
double distKernel(double d, Vector & params, int n)
{
  switch (n)
    {
    case 0: return params(0) * std::exp( -d / (2.0*std::pow(params(1),2)));
    case 1: return std::exp( -d / (2.0*std::pow(params(1),2)));
    case 2: return params(0) * d / std::pow(params(1),3) * std::exp( -d / (2.0*std::pow(params(1),2)));
    default: std::cout << "\n[*] UNDEFINED DERIVATIVE\n"; return 0.0;
    }
}
*/

// Define squared exponential kernel between two points
double kernel(Matrix & x, Matrix & y, Vector & params, int n)
{
  switch (n)
    {
    case 0: return std::exp( -(x-y).squaredNorm() / (2.0*std::pow(params(0),2)));
    case 1: return (x-y).squaredNorm() / std::pow(params(0),3) * std::exp( -(x-y).squaredNorm() / (2.0*std::pow(params(0),2)));
    default: std::cout << "\n[*] UNDEFINED DERIVATIVE\n"; return 0.0;
    }
}

// Define squared exponential kernel provided a squared distance as input
double distKernel(double d, Vector & params, int n)
{
  switch (n)
    {
    case 0: return std::exp( -d / (2.0*std::pow(params(0),2)));
    case 1: return d / std::pow(params(0),3) * std::exp( -d / (2.0*std::pow(params(0),2)));
    default: std::cout << "\n[*] UNDEFINED DERIVATIVE\n"; return 0.0;
    }
}

// Specify the true target function
double targetFunc(double x)
{
  return std::sin(3.0*(x-0.1))*(0.5-(x-0.1))*15.0;
}

// Define function for uniform sampling
Matrix sampleUnif(double a=0.0, double b=1.0, int N=1)
{
  return (b-a)*(Eigen::MatrixXd::Random(N,1) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,1)) + a*Eigen::MatrixXd::Ones(N,1);
}

// Example using GP class for regression
int main(int argc, char const *argv[])
{
  using std::cout;
  using std::endl;
  using GP::GaussianProcess;
  using GP::linspace;

  // Aliases for timing functions with chrono
  using std::chrono::high_resolution_clock;
  using time = high_resolution_clock::time_point;
  
  // Set random seed
  std::srand(static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));
  //std::srand(static_cast<unsigned int>(std::time(0)));

  // Initialize Gaussian process model
  GaussianProcess model(1);

  // Specify observation data
  int obsCount = 250;
  Matrix x = sampleUnif(0.0, 1.0, obsCount);
  Matrix y;
  auto noiseLevel = 0.05;
  auto noise = Eigen::VectorXd::Random(obsCount) * noiseLevel;
  //x.resize(obsCount, 1);
  //x = linspace(0.0, 1.0, obsCount);
  y.resize(obsCount, 1);
  y = x.unaryExpr(std::ptr_fun(targetFunc)) + noise;
  model.setObs(x,y);

  // Fix noise level
  model.setNoise(std::pow(noiseLevel, 2));
  //model.setNoise(noiseLevel);
  //model.setNoise(std::sqrt(noiseLevel));
  
  // Define initial kernel parameters
  //Vector params(2);
  //params << 1.0, 1.0;
  Vector params(1);
  params << 1.0;


  // Define kernel for GP model
  model.setKernel( std::make_unique<GP::kernelfn>(kernel) , params );
  model.setDistKernel( std::make_unique<GP::distkernelfn>(distKernel) , params );

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
