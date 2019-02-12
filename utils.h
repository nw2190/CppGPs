#ifndef _UTILS_H
#define _UTILS_H
#include <iostream>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

// Declare namespace for utils
namespace utils {

  // Define aliases with using declarations
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using Eigen::VectorXcd;

  // Generate equally spaced points on an interval (Eigen::Vector)
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, 1> linspace(T a, T b, int N) { return Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(N, a, b); };

  
  // Construct Toeplitz matrix from column (symmetric)
  Matrix buildToep(const Vector & col, const Vector & row);

  // Construct Toeplitz matrix from column (symmetric)
  Matrix buildToep(const Vector & col);

  // Embed Toeplitz matrix into column of circulant matrix (general)  
  void embedToep(Vector & toepCol, Vector & toepRow, VectorXcd & eigVals);

  // Embed Toeplitz matrix into column of circulant matrix (symmetric)
  void embedToep(Vector & toepCol, VectorXcd & eigVals);

  
  // Define matvec product for circular matrix using FFT (non-symmetric / complex eigVals)
  void circMatVec(const VectorXcd & eigVals, Vector & x, Vector & result);

  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Vector toepMatVec(Vector & col, Vector & row, Vector & x, bool sym=false);
  
  // Define matvec product for Toeplitz matrix using FFT (symmetric)
  Vector toepMatVec(Vector & col, Vector & x);

  // NOTE: the "fast" version avoids extra function calls and
  // is marginally faster than the more pedagogical "toepMatVec"
  
  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Vector fastToepMatVec(Vector & toepCol, Vector & toepRow, Vector & x);

  // Define matvec product for Toeplitz matrix using FFT (symmetric)
  Vector fastToepMatVec(Vector & col, Vector & x);


  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Matrix fastToepMatMat(Vector & toepCol, Vector & toepRow, Matrix & X);

  // Define matvec product for Toeplitz matrix using FFT (non-symmetric)
  Matrix fastToepMatMat(Vector & toepCol, Matrix & X);
  
  // Compute Lanczos vectors for matrix-vector pair (A,v)
  void Lanczos(Matrix & A, Vector & v, Matrix & Q, Matrix & T, int N=0);

  
};
#endif
