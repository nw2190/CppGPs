#ifndef _UTILS_H
#define _UTILS_H
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

// Declare namespace for utils
namespace utils {

  // Define aliases with using declarations
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using Eigen::VectorXcd;

  // Aliases for timing functions with chrono
  using std::chrono::high_resolution_clock;
  using time = high_resolution_clock::time_point;

  float getTime(time start, time end)
  {
    return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() / 1000000.0);
  }

  
  // Generate equally spaced points on an interval (Eigen::Vector)
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, 1> linspace(T a, T b, int N)
  { return Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(N, a, b); }

  
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


  
  // NOTE: the "fast" version avoids extra function calls and
  // is marginally faster than the more pedagogical "toepMatVec"
  
  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Vector fastToepMatVec(Vector & toepCol, Vector & toepRow, Vector & x)
  {
    auto n = static_cast<int>(toepCol.rows());
    Vector result(2*n);

    // Embed RHS into 2n-vector with trailing zeros
    Vector xembed = Eigen::VectorXd::Zero(2*n);
    xembed.head(n) = x;

    // Specify first column of circulant matrix embedding
    Vector embedCol(2*n);
    embedCol.head(n) = toepCol;
    embedCol.tail(n-1) = toepRow.tail(n-1).reverse();

    // Compute eigenvalues of cirulant matrix
    Eigen::FFT<double> fft;
    VectorXcd eigVals(2*n), freqvec(2*n);
    fft.fwd(eigVals, embedCol);

    // Apply component-wise multiplication in frequency space
    fft.fwd(freqvec, xembed);
    freqvec = eigVals.cwiseProduct(freqvec);
    fft.inv(result,freqvec);

    // Return first n values corresponding to the original system
    return result.head(n);

  };
  
  // Define matvec product for Toeplitz matrix using FFT (symmetric)
  Vector fastToepMatVec(Vector & col, Vector & x) { return fastToepMatVec(col, col, x); };



  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Vector diagnosticFastToepMatVec(Vector & toepCol, Vector & x)
  {
    auto n = static_cast<int>(toepCol.rows());
    Vector xembed = Eigen::VectorXd::Zero(2*n);
    xembed.head(n) = x;
    Vector result(2*n);

    // Specify first column of circulant matrix embedding
    Vector embedCol(2*n);
    embedCol.head(n) = toepCol;
    embedCol.tail(n-1) = toepCol.tail(n-1).reverse();

    
    // Compute eigenvalues of cirulant matrix
    Eigen::FFT<double> fft;
    VectorXcd eigVals(2*n);
    VectorXcd freqvec(2*n);

    std::cout << "\nMatVec Time Diagnostics:\n";
    
    time start = high_resolution_clock::now();
    fft.fwd(eigVals, embedCol);
    time end = high_resolution_clock::now();
    auto duration = getTime(start, end);
    std::cout << " - fft.fwd(): \t" << duration << std::endl;

    start = high_resolution_clock::now();
    fft.fwd(freqvec, xembed);
    end = high_resolution_clock::now();
    duration = getTime(start, end);
    std::cout << " - fft.fwd(): \t" << duration << std::endl;

    start = high_resolution_clock::now();    
    freqvec = eigVals.cwiseProduct(freqvec);
    end = high_resolution_clock::now();
    duration = getTime(start, end);
    std::cout << " - cwise(): \t" << duration << std::endl;

    start = high_resolution_clock::now();    
    fft.inv(result,freqvec);
    end = high_resolution_clock::now();
    duration = getTime(start, end);
    std::cout << " - fft.inv(): \t" << duration << std::endl << std::endl;

    return result.head(n);

  };
  

};
#endif
