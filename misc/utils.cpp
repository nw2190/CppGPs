#include <iostream>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <boost/range/irange.hpp>
#include <unsupported/Eigen/FFT>
#include "utils.h"

// Define aliases with using declarations
using Matrix = utils::Matrix;
using Vector = utils::Vector;
using VectorXcd = utils::VectorXcd;

// Construct Toeplitz matrix from column (symmetric)
Matrix utils::buildToep(const Vector & col, const Vector & row)
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
Matrix utils::buildToep(const Vector & col) { return buildToep(col, col); };

// Embed Toeplitz matrix into column of circulant matrix (general)  
void utils::embedToep(Vector & toepCol, Vector & toepRow, VectorXcd & eigVals)
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
void utils::embedToep(Vector & toepCol, VectorXcd & eigVals) { embedToep(toepCol, toepCol, eigVals); };


// Define matvec product for circular matrix using FFT (non-symmetric / complex eigVals)
void utils::circMatVec(const VectorXcd & eigVals, Vector & x, Vector & result)
{
  VectorXcd freqvec;
  Eigen::FFT<double> fft;
  fft.fwd(freqvec,x);
  freqvec = eigVals.cwiseProduct(freqvec);
  fft.inv(result,freqvec);
};


// Define matvec product for Toeplitz matrix using FFT (non-symmetric)
Vector utils::toepMatVec(Vector & col, Vector & row, Vector & x, bool sym)
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
Vector utils::toepMatVec(Vector & col, Vector & x) { return toepMatVec(col, col, x, true); };



// NOTE: the "fast" version avoids extra function calls and
// is marginally faster than the more pedagogical "toepMatVec"

// Define matvec product for Toeplitz matrix using FFT (non-symmetric)
Vector utils::fastToepMatVec(Vector & toepCol, Vector & toepRow, Vector & x)
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
Vector utils::fastToepMatVec(Vector & col, Vector & x) { return fastToepMatVec(col, col, x); };



// Define matvec product for Toeplitz matrix using FFT (non-symmetric)
Matrix utils::fastToepMatMat(Vector & toepCol, Vector & toepRow, Matrix & X)
{
  auto n = static_cast<int>(toepCol.rows());
  auto m = static_cast<int>(X.cols());
  Matrix result(n,m);
  for (auto i : boost::irange(0,m))
    {
      Vector X_i = X.col(i);
      result.col(i) = fastToepMatVec(toepCol, toepRow, X_i);
    }
  return result;
}

// Define matvec product for Toeplitz matrix using FFT (non-symmetric)
Matrix utils::fastToepMatMat(Vector & toepCol, Matrix & X) { return fastToepMatMat(toepCol, toepCol, X); };



// Compute Lanczos vectors for matrix-vector pair (A,v)
void utils::Lanczos(Matrix & A, Vector & v, Matrix & Q, Matrix & T, int N)
{
  // Use full decomposition if step count N is not specified
  auto n = static_cast<int>(A.rows());
  if (N == 0)
    N = n;

  // Initialize the algorithm's vectors and parameters
  Vector q = v/v.norm();
  auto alpha = static_cast<double>( (A*q).dot(q) );
  Vector r = A*q - alpha*q;
  Q.col(0) = q;
  T(0,0) = alpha;
  Vector u;

  // Define orthogonality and vector norm tolerances
  //double epsilon = 1.0e-15;
  //double eta = std::pow(epsilon, 0.75);
  double tolerance = 0.00001;

  for (auto j : boost::irange(1,N))
    {
      auto beta = static_cast<double>(r.norm());    
      if (beta < tolerance)
        {
          std::cout << "\nStopped Early\n";
          Q.resize(n,j);
          break;
        }

      // Basic implementation
      //v_k = vt_k/beta;
      //alpha = v_k.transpose().dot(A*v_k);
      //vt_k = A*v_k - alpha*v_k - beta*Q.col(k-1);

      // Improved stability implementation 
      q = r/beta;
      u = A*q - beta*Q.col(j-1);
      alpha = q.transpose().dot(u);
      r = u - alpha*q;

      // Complete re-orthogonalization [ roughly ~ O(n^3) for combined iterations ]
      r = r - Q*(Q.transpose()*r);

      /*
      // Check if additional re-orthogonalization is necessary
      for (auto k : boost::irange(0,j))
        {
          if ( std::abs((r/r.norm()).dot(Q.col(k))) > eta )
            {
              //std::cout << "Reorthogonalizing..." << std::endl;
              r = r/r.norm();
              r = r - Q*(Q.transpose()*r);
              break;
            }
        }
      */

      // Assign Lanczos vector and tridiagonal entries
      Q.col(j) = q;
      T(j,j) = alpha;
      T(j,j-1) = T(j-1,j) = beta;

    }

};
