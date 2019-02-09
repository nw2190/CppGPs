// main.cpp -- test minimize function definitions
#include <iostream>
#include <chrono>
#include <boost/range/irange.hpp>
#include "minimize.h"

using minimize::Matrix;
using minimize::Vector;
using minimize::VectorXcd;
//using minimize::VectorXld;


// Define function/derivative evaluations
double func(Vector X, Vector & D)
{
  D(0) = (-200.0*X(0)*(std::pow(X(0) - 0.2, 2) + std::pow(X(1) + 0.4, 2)) + (200.0*X(0) - 40.0)*(std::pow(X(0), 2) + 0.1*std::pow(X(1), 4) + 1))/std::pow(std::pow(X(0), 2) + 0.1*std::pow(X(1), 4) + 1, 2);

  D(1) = (-40.0*std::pow(X(1), 3)*(std::pow(X(0) - 0.2, 2) + std::pow(X(1) + 0.4, 2)) + (200.0*X(1) + 80.0)*(std::pow(X(0), 2) + 0.1*std::pow(X(1), 4) + 1))/std::pow(std::pow(X(0), 2) + 0.1*std::pow(X(1), 4) + 1, 2);

  return 100.0*(std::pow(X(0) - 0.2, 2) + std::pow(X(1) + 0.4, 2))/(std::pow(X(0), 2) + 0.1*std::pow(X(1), 4) + 1);
}


// Define function for retrieving time from chrono
float getTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end)
{
  return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);
};

// Define function for uniform sampling
Matrix sampleUnif(double a=0.0, double b=1.0, int N=1)
{
  return (b-a)*(Eigen::MatrixXd::Random(N,1) * 0.5 + 0.5*Eigen::MatrixXd::Ones(N,1)) + a*Eigen::MatrixXd::Ones(N,1);
};

// Declare undefined class for checking deduced types
template<typename T>
class CheckType;

// Example using GP class for regression
int main(int argc, char const *argv[])
{

  // Aliases for convenience
  using std::cout;
  using std::endl;

  using minimize::cg_minimize;
  
  // Aliases for timing functions with chrono
  //using std::chrono::high_resolution_clock;
  //using time = high_resolution_clock::time_point;
  
  // Set random seed
  //std::srand(static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));

  // Specify precision of minimization algorithm
  double SIG = 0.1;
  double EXT = 3.0;
  double INT = 0.01;
  
  Vector X(2);
  X << 0.0, 0.0;
  Vector D(2);

  int length = 1000;
  cg_minimize(X, std::make_unique<minimize::minimizefn>(func), D, length, SIG, EXT, INT);

  // Display computed solution
  cout << X << endl;

  // Display function/derivative values
  double func_val = func(X,D);
  cout << func_val << " ,  [ " << D.transpose() << " ] " << endl;
  
  return 0;
}
