// main.cpp -- test utility function definitions
#include <iostream>
#include <cmath>
#include <string>
#include <array>
#include <vector>
#include <memory>
#include <random>
#include <ctime>
#include <fstream>
#include <chrono>
#include <boost/range/irange.hpp>
#include "utils.h"

using utils::Matrix;
using utils::Vector;
using utils::VectorXcd;

// Declare undefined class for checking deduced types
template<typename T>
class CheckType;

// Example using GP class for regression
int main(int argc, char const *argv[])
{
  using std::cout;
  using std::endl;
  using utils::linspace;
  using utils::vlinspace;
  //using utils::Vector;
  
  using utils::buildToep;
  using utils::embedToep;
  //using utils::circEigVals;
  using utils::circMatVec;
  using utils::toepMatVec;

  // Aliases for timing functions with chrono
  using std::chrono::high_resolution_clock;
  using time = high_resolution_clock::time_point;
  
  // Set random seed
  std::srand(static_cast<unsigned int>(std::time(0)));

  int N = 10000;
  Vector mesh = linspace(0.0,1.0,N);
  Vector x = Eigen::VectorXd::Random(N);

  // Build Toeplitz matrices for direct evaluation
  time start = high_resolution_clock::now();
  Matrix toep = buildToep(mesh.reverse());
  Matrix toep2 = buildToep(mesh.reverse(), mesh);
  time end = high_resolution_clock::now();
  auto buildDuration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);

  start = high_resolution_clock::now();
  Vector toep_col = toep.col(0);
  auto fftResult = toepMatVec(toep_col, x);
  end = high_resolution_clock::now();
  auto toepDuration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);

  
  start = high_resolution_clock::now();
  auto directResult = toep*x;
  end = high_resolution_clock::now();
  auto directDuration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);


  start = high_resolution_clock::now();
  Vector toep2_col = toep2.col(0);
  Vector toep2_row = toep2.row(0);
  auto fftResult2 = toepMatVec(toep2_col, toep2_row, x);
  end = high_resolution_clock::now();
  auto toepDuration2 = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);


  start = high_resolution_clock::now();
  auto directResult2 = toep2*x;
  end = high_resolution_clock::now();
  auto directDuration2 = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);


  // Display errors for symmetric and non-symmetric cases
  cout << "Symmetric Case:" << endl;
  cout << (directResult - fftResult).squaredNorm() << endl;

  cout << "\nNon-Symmetric Case:" << endl;
  cout << (directResult2 - fftResult2).squaredNorm() << endl;


  // Display evaluation times for symmetric and non-symmetric cases
  cout << endl << "\nSymmetric Case:" << endl;
  cout << "Direct: " << directDuration << " s " << "\tFFT: " << toepDuration << " s" << endl;

  cout << "\nNon-Symmetric Case:" << endl;
  cout << "Direct: " << directDuration2 << " s " << "\tFFT: " << toepDuration2 << " s" << endl;

  cout << endl << "[ Build duration: " << buildDuration << " ]" << endl;
  
  return 0;
}
