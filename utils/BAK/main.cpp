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
  using utils::circEigVals;
  using utils::circMatVec;
  using utils::toepMatVec;
  
  // Set random seed
  std::srand(static_cast<unsigned int>(std::time(0)));

  int N = 5;
  //auto mesh = vlinspace<double>(0.0,1.0,N);
  Vector mesh = linspace(0.0,1.0,N);
  //for (auto i : boost::irange(0,N))
  //  cout << mesh[i] << ((i<N-1) ? " " : "\n");

  Matrix toep = buildToep(mesh.reverse());
  //cout << toep << endl;

  /*
  Vector circ = embedToep(toep.col(0));
  //cout << circ << endl;
  Vector eigVals = circEigVals(circ);
  //cout << eigVals << endl;
  
  Vector x = mesh;
  Vector xembed(2*N);
  xembed.head(N) = x;
  xembed.tail(N) = Eigen::VectorXd::Zero(N);
  
  auto circResult = circMatVec(eigVals, xembed);
  //cout << circResult << endl << endl;
  */

  Vector x = mesh;

  Vector toep_col = toep.col(0);
  auto fftResult = toepMatVec(toep_col, x);
  cout << fftResult << endl << endl;
    
  auto directResult = toep*x;
  cout << directResult << endl;

  Matrix toep2 = buildToep(mesh.reverse(), mesh);
  Vector toep2_col = toep2.col(0);
  Vector toep2_row = toep2.row(0);
  auto fftResult2 = toepMatVec(toep2_col, toep2_row, x);
  cout << endl << endl << fftResult2 << endl << endl;
    
  auto directResult2 = toep2*x;
  cout << directResult2 << endl;
  
  //std::vector<float> timevec;
  //auto timevec = static_cast<std::vector<double> >(mesh);
  //std::vector<std::complex<double> > freqvec;
  auto timevec = mesh;
  VectorXcd freqvec;
  //auto output = utils::customFFT<double>(timevec, freqvec);
  //auto output = utils::circMatVec(timevec, freqvec);
  //for (auto i : boost::irange(0,N))
  //  cout << output[i] << ((i<N-1) ? " " : "\n");
  
  return 0;
}
