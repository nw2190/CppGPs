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
  
  // Generate equally spaced points on an interval (Eigen::Vector)
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, 1> linspace(T a, T b, int N)
  { return Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(N, a, b); }

  // Generate equally spaced points on an interval (std::vector)
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
  Matrix buildToep(const Vector & col, const Vector & row)
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
  Matrix buildToep(const Vector & col) { return buildToep(col, col); };

  
  // Embed Toeplitz matrix into column of circulant matrix (general)  
  void embedToep(Vector & toepCol, Vector & toepRow, VectorXcd & eigVals)
  {
    // Specify first column of circulant matrix embedding
    auto n = static_cast<int>(toepCol.rows());
    Vector embedCol(2*n);
    embedCol.head(n) = toepCol;
    embedCol.tail(n-1) = toepRow.tail(n-1).reverse();

    // Compute eigenvalues of cirulant matrix
    Eigen::FFT<double> fft;
    fft.fwd(eigVals, embedCol);
  };
    
  // Embed Toeplitz matrix into column of circulant matrix (symmetric)
  void embedToep(Vector & toepCol, VectorXcd & eigVals) { embedToep(toepCol, toepCol, eigVals); };

  
  
  // Define matvec product for circular matrix using FFT (non-symmetric / complex eigVals)
  void circMatVec(const VectorXcd & eigVals, Vector & x, Vector & result)
  {
    VectorXcd freqvec;
    Eigen::FFT<double> fft;
    fft.fwd(freqvec,x);
    freqvec = eigVals.cwiseProduct(freqvec);
    fft.inv(result,freqvec);
  };
  

  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Vector toepMatVec(Vector & col, Vector & row, Vector & x, bool sym=false)
  {
    auto n = static_cast<int>(col.rows());
    Vector xembed = Eigen::VectorXd::Zero(2*n);
    xembed.head(n) = x;
    VectorXcd eigVals(2*n);
    Vector result(2*n);
    embedToep(col, row, eigVals);

    // Note: Should be possible to optimize this better using
    // the fact that eigVals is real in the symmetric case
    if (sym)
        circMatVec(eigVals.real(), xembed, result);
    else
        circMatVec(eigVals, xembed, result);

    return result.head(n);

  };
  
  // Define matvec product for Toeplitz matrix using FFT (symmetric)
  Vector toepMatVec(Vector & col, Vector & x) { return toepMatVec(col, col, x, true); };


};
#endif
