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
  using utils::diagnosticFastToepMatVec;

  using utils::Lanczos;


  // Aliases for timing functions with chrono
  using std::chrono::high_resolution_clock;
  using time = high_resolution_clock::time_point;
  
  // Set random seed
  std::srand(static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));

  /*
  // Define values for Toeplitz column
  int N = 15000;
  Vector mesh = linspace(0.0,1.0,N);
  mesh = mesh + sampleUnif(-0.1, 0.1, N);

  // Define vector for matvec product
  Vector x = Eigen::VectorXd::Random(N);

  
  // Build Toeplitz matrices for direct evaluation
  time start = high_resolution_clock::now();
  Matrix toep = buildToep(mesh.reverse());
  Matrix toep2 = buildToep(mesh.reverse(), mesh);
  time end = high_resolution_clock::now();
  auto buildDuration = getTime(start, end);


  // Evaluate using FFT (symmetric)
  Vector toep_col = toep.col(0);
  start = high_resolution_clock::now();
  //auto fftResult = toepMatVec(toep_col, x);
  auto fftResult = fastToepMatVec(toep_col, x);
  end = high_resolution_clock::now();
  auto toepDuration = getTime(start, end);


  // Evaluate using direct matvec (symmetric)  
  // NOTE: the use of ".noalias()" forces lazy evaluation
  // and allows us to retrieve the actual computation time
  Vector directResult;  
  start = high_resolution_clock::now();
  //auto directResult = toep*x;
  directResult.noalias() = toep*x;
  end = high_resolution_clock::now();
  auto directDuration = getTime(start, end);


  // Evaluate using FFT (non-symmetric)  
  Vector toep2_col = toep2.col(0);
  Vector toep2_row = toep2.row(0);
  start = high_resolution_clock::now();
  //auto fftResult2 = toepMatVec(toep2_col, toep2_row, x);
  auto fftResult2 = fastToepMatVec(toep2_col, toep2_row, x);
  end = high_resolution_clock::now();
  auto toepDuration2 = getTime(start, end);

  // Evaluate using direct matvec (non-symmetric)  
  Vector directResult2;
  start = high_resolution_clock::now();
  //auto directResult2 = toep2*x;
  directResult2.noalias() = toep2*x;
  end = high_resolution_clock::now();
  auto directDuration2 = getTime(start, end);


  // Display errors for symmetric and non-symmetric cases
  cout << "\nAbsolute Error (Symmetric):" << endl;
  cout << (directResult - fftResult).norm() << endl;

  cout << "\nAbsolute Error (Non-Symmetric):" << endl;
  cout << (directResult2 - fftResult2).norm() << endl;

  // Display evaluation times for symmetric and non-symmetric cases
  cout << endl << "\nEvaluation Time (Symmetric):" << endl;
  cout << "Direct: " << directDuration << " s " << "\tFFT: " << toepDuration << " s" << endl;

  cout << "\nEvaluation Time (Non-Symmetric):" << endl;
  cout << "Direct: " << directDuration2 << " s " << "\tFFT: " << toepDuration2 << " s" << endl;

  // Display build time to construct dense Toeplitz matrices
  cout << endl << "[ Build duration: " << buildDuration << " ]" << endl << endl;

  */

  // Define values for Toeplitz column
  int N = 200;
  Vector mesh = linspace(0.0,1.0,N);
  mesh = mesh + sampleUnif(-0.1, 0.1, N);

  // Build Toeplitz matrices for direct evaluation
  Matrix A = buildToep(mesh.reverse());

  // Define vector for multiplication
  Vector v = Eigen::VectorXd::Random(N);
  v.normalize();
  
  // Define matrices to store decomposition
  Matrix V(N,N);
  Matrix T(N,N);

  time start = high_resolution_clock::now();
  Lanczos(A, v, V, T);
  time end = high_resolution_clock::now();
  auto lanczosTime = getTime(start, end);
  
  cout << "Lanczos Decomposition Error:\n";
  cout << (A*V - V*T).norm() << endl;

  cout << "Lanczos Evaluation Time:\n";
  cout << lanczosTime << " s" << endl;


  
  return 0;
}
