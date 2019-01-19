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
  
  // Set random seed
  std::srand(static_cast<unsigned int>(std::time(0)));

  int N = 5000;
  Vector mesh = linspace(0.0,1.0,N);
  Vector x = Eigen::VectorXd::Random(N);

  Matrix toep = buildToep(mesh.reverse());
  Vector toep_col = toep.col(0);
  auto fftResult = toepMatVec(toep_col, x);
  auto directResult = toep*x;

  Matrix toep2 = buildToep(mesh.reverse(), mesh);
  Vector toep2_col = toep2.col(0);
  Vector toep2_row = toep2.row(0);
  auto fftResult2 = toepMatVec(toep2_col, toep2_row, x);
  auto directResult2 = toep2*x;

  // Display errors for symmetric and non-symmetric cases
  cout << "Symmetric Case:" << endl;
  cout << (directResult - fftResult).squaredNorm() << endl;

  cout << "\nNon-Symmetric Case:" << endl;
  cout << (directResult2 - fftResult2).squaredNorm() << endl;
  
  return 0;
}
