// main.cpp -- test utility function definitions
#include <iostream>
#include <chrono>
#include <boost/range/irange.hpp>
#include "utils.h"

using utils::Matrix;
using utils::Vector;
using utils::VectorXcd;


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

// Example using GP class for regression
int main(int argc, char const *argv[])
{

  // Aliases for convenience
  using std::cout;
  using std::endl;

  using utils::linspace;
  using utils::buildToep;
  using utils::toepMatVec;
  using utils::fastToepMatVec;
  using utils::fastToepMatMat;
  using utils::Lanczos;


  // Aliases for timing functions with chrono
  //using std::chrono::high_resolution_clock;
  //using time = high_resolution_clock::time_point;
  
  // Set random seed
  //std::srand(static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));


  // Define values for Toeplitz column
  int N = 100;
  Vector mesh = linspace(0.0,1.0,N);
  mesh = mesh + sampleUnif(-0.1, 0.1, N);

  Vector params(1);
  params << 0.1;
  Vector toepCol(N);

  for ( auto i : boost::irange(0,N) )
    toepCol(i) = distKernel( (mesh.row(0) - mesh.row(i)).squaredNorm() ,params,0);

  // Build Toeplitz matrices for direct evaluation
  //Matrix K = buildToep(mesh.reverse());
  Matrix K = buildToep(toepCol);
  K = K - 1.0*Matrix::Identity(N,N);
  cout << "\nMatrix K:\n" << K << endl;


  Eigen::JacobiSVD<Matrix> svd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Matrix U = svd.matrixU();
  Matrix V = svd.matrixV();
  Matrix singularValues = svd.singularValues();

  //cout << "\nMatrix U:\n" << U << endl;
  //cout << "\nMatrix V:\n" << V << endl;
  cout << "\nSingular Values:\n" << singularValues.transpose() << endl;
  
  return 0;
}
