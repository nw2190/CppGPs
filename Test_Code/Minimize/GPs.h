#ifndef _GPS_H
#define _GPS_H
#include <iostream>
#include <array>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include "minimize.h"

// Declare separate namespace for
// Gaussian process definitions
namespace GP {

  // Define PI using arctan function
  static const double PI = std::atan(1)*4;

  // Define aliases with using declarations
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  // Define kernel and kernel_pointer aliases
  using kernelfn = double (*)(Matrix&, Matrix&, Vector&, int);
  using kernelptr = std::unique_ptr<kernelfn>;
  using distkernelfn = double (*)(double, Vector&, int);
  using distkernelptr = std::unique_ptr<distkernelfn>;


  // Define class for Gaussian processes
  class GaussianProcess : public minimize::GradientObj
  {    
  public:
    // Constructors
    GaussianProcess() : dimIn(1) { }
    GaussianProcess(int din) : dimIn(din) { }
    // Copy Constructor
    GaussianProcess(const GaussianProcess & m) :
      useDistKernel(m.useDistKernel) ,
      noiseLevel(m.noiseLevel),
      fixedNoise(m.fixedNoise),
      kernel( (m.kernel) ?  std::make_unique<kernelfn>(kernelfn(*(m.kernel))) : nullptr ) ,
      distKernel( (m.distKernel) ? std::make_unique<distkernelfn>(distkernelfn(*(m.distKernel))) : nullptr) ,
      paramCount(m.paramCount),
      obsX(m.obsX) ,
      obsY(m.obsY)
      { N = static_cast<int>(obsX.rows()); }
      //{ std::cout << "\nCOPY\n"; N = static_cast<int>(obsX.rows()); }


    // Get and show methods
    int getDim() { return dimIn; }
    void showObs();
    void showCov(int prec=5);
    double evalNLML(const Vector & p); //, Matrix & alpha);
    void evalDNLML(const Vector & p, Vector & g); //, Matrix & alpha);
    double getNLML() { return NLML; }
    Matrix getCov() { return K; }
    Matrix getPredMean() { return predMean; }
    //Matrix getPredVar() { return predCov.diagonal() + noiseLevel*noiseLevel*Eigen::VectorXd::Ones(predMean.size()); }
    Matrix getPredVar() { return predCov.diagonal() + noiseLevel*Eigen::VectorXd::Ones(predMean.size()); }
    Matrix getSamples(int count=10);
    decltype(auto) getParams() { return kernelParams; }
    double getNoise() { return noiseLevel; }
    
    // Set methods
    void setObs(Matrix & x, Matrix & y) { obsX = x; obsY = y; N = static_cast<int>(x.rows()); }
    void setKernel(kernelptr k, Vector p) { kernelParams = p; paramCount = p.size(); kernel = std::move(k); }
    void setDistKernel(distkernelptr k, Vector p) { useDistKernel = true; kernelParams = p; paramCount = p.size(); distKernel = std::move(k); }
    void setPred(Matrix & px) { predX = px; }

    // NOISE
    void setNoise(double noise) { fixedNoise = true; noiseLevel = noise; }
    //void setNoise(double noise) { fixedNoise = true; noiseLevel = 0.0; }
    
    // Compute methods
    void computeCov(Vector & p, bool useDistMat);
    void computeCov(bool useDistMat);
    void computeChol(double noise);
    void computeChol();
    void predict();
    void fitModel();

    // Precompute distance matrix for optimization procedure
    void computeDistMat();
    
    // Define method for superclass "GradientObj" used by minimization algorithm
    void computeValueAndGradient(Vector X,double & val, Vector & D) { val = evalNLML(X); evalDNLML(X,D); };

    
  private:
    // Private member functions
    double evalKernel(Matrix&& x1, Matrix&& x2, int deriv=0) { return (*kernel)(x1, x2, kernelParams, deriv); }
    void computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p, int deriv, bool useDistMat);
    void computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p, bool useDistMat);
    void computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, bool useDistMat);

    // Status variables
    bool useDistKernel = false;
    int dimIn;
    int N = 0;

    // Kernel and covariance matrix
    double noiseLevel = 0.00001;
    bool fixedNoise = false;
    double jitter = 1e-7;
    kernelptr kernel;    
    distkernelptr distKernel;
    Vector kernelParams;
    int paramCount;
    Matrix K;
    Eigen::LLT<Matrix> cholesky;

    // Store squared distance matrix and alpha for NLML/DNLML calculations
    Matrix distMatrix;
    Matrix _alpha;
    
    // Observation data
    Matrix obsX; // size: N x dimIn
    Matrix obsY; // size: N x 1
    
    // Prediction data
    Matrix predX;
    Matrix predMean;
    Matrix predCov;
    double NLML = 0.0;

  };

  // Define linspace function for generating
  // equally spaced points on an interval
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> linspace(T a, T b, int N)
  { return Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(N, a, b); }

    
  
};


#endif
