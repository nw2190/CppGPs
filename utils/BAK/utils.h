#ifndef _UTILS_H
#define _UTILS_H
#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

// Declare namespace for utils
namespace utils {
  // Define PI using arctan function
  static const double PI = std::atan(1)*4;

  // Define aliases with using declarations
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using Eigen::VectorXcd;
  //using Vectord = Eigen::Matrix<double, Eigen::Dynamic, 1>;
  //using Vectorc = Eigen::Matrix<std::complex<double> , Eigen::Dynamic, 1>;
  
  // Define linspace function for generating
  // equally spaced points on an interval
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, 1> linspace(T a, T b, int N)
  { return Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(N, a, b); }

  template<typename T>
  std::vector<T> vlinspace(T a, T b, int n)
  {
    std::vector<T> array;
    double step = (b-a) / (n-1);
    for (auto i : boost::irange(0,n))
        array.push_back(a + i*step);
    return array;
  };
  


  // Construct Toeplitz matrix from column (symmetric)
  Matrix buildToep(Vector col, Vector row)
  {
    auto n = static_cast<int>(col.rows());
    Matrix toep(n,n);
    for (auto i : boost::irange(0,n))
      {
        toep.diagonal(i) = Eigen::VectorXd::Ones(n-i)*row[i];
        toep.diagonal(-i) = Eigen::VectorXd::Ones(n-i)*col[i];
      }
    return toep;
  };

  // Construct Toeplitz matrix from column (symmetric)
  Matrix buildToep(Vector col) { return buildToep(col, col); };

  
  // Embed Toeplitz matrix into column of circulant matrix (general)  
  Vector embedToep(Vector toepCol, Vector toepRow)
  {
    auto n = static_cast<int>(toepCol.rows());
    Vector circ(2*n);
    circ.head(n) = toepCol;
    circ.tail(n-1) = toepRow.tail(n-1).reverse();
    return circ;
  }
    
  // Embed Toeplitz matrix into column of circulant matrix (symmetric)
  Vector embedToep(Vector toepCol) { return embedToep(toepCol, toepCol); };

  
  // Compute eigenvalues of circulant matrix
  VectorXcd circEigVals(Vector col)
  {
    VectorXcd eigVals;
    Eigen::FFT<double> fft;
    fft.fwd(eigVals, col);
    return eigVals;
    //return eigVals.real();
  };
  
  // Define matvec product for circular matrix using FFT (symmetric / real eigVals)
  Vector circMatVecReal(Vector eigVals, Vector x)
  {
    VectorXcd freqvec;
    Eigen::FFT<double> fft;
    fft.fwd(freqvec,x);
    freqvec = freqvec.cwiseProduct(eigVals);
    fft.inv(x,freqvec);
    return x;
  };

  // Define matvec product for circular matrix using FFT (non-symmetric / complex eigVals)
  Vector circMatVec(VectorXcd eigVals, Vector x)
  {
    VectorXcd freqvec;
    VectorXcd result;
    Eigen::FFT<double> fft;
    fft.fwd(freqvec,x);
    freqvec = freqvec.cwiseProduct(eigVals);
    fft.inv(result,freqvec);
    return result.real();
  };
  

  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Vector toepMatVec(Vector col, Vector row, Vector x, bool sym=false)
  {
    auto n = static_cast<int>(col.rows());
    Vector xembed = Eigen::VectorXd::Zero(2*n);
    xembed.head(n) = x;
    if (sym)
      return circMatVecReal(circEigVals(embedToep(col,row)).real(), xembed).head(n);
    else
      return circMatVec(circEigVals(embedToep(col,row)), xembed).head(n);
  };
  
  // Define matvec product for Toeplitz matrix using FFT (symmetric)
  Vector toepMatVec(Vector col, Vector x)
  {
    return toepMatVec(col, col, x, true);
  };

  /*
  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Vector toepMatVec(Vector col, Vector row, Vector x, bool symmetric=false)
  {
    auto n = static_cast<int>(col.rows());
    Vector xembed = Eigen::VectorXd::Zero(2*n);
    xembed.head(n) = x;
    auto result = circMatVec(circEigVals(embedToep(col,row)), xembed);
    return result.head(n);
  };
  
  // Define matvec product for Toeplitz matrix using FFT (symmetric)
  Vector toepMatVec(Vector col, Vector x)
  {
    auto n = static_cast<int>(col.rows());
    Vector xembed = Eigen::VectorXd::Zero(2*n);
    xembed.head(n) = x;
    auto result = circMatVecReal(circEigVals(embedToep(col)).real(), xembed);
    return result.head(n);
  };
  */

};
#endif
