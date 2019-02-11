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


  // Define abstract class for covariance kernels

  /*
  class Kernel 
  {    
  public:
    // Constructors
    //Kernel() : kernelParams(Vector(0)) , paramCount(0) { };
    //Kernel(Vector p, int n);
    virtual void computeCov(Matrix & K, Matrix & D, Vector & params, int deriv);
    
    //int getParamCount() { return paramCount; } ;
    //Vector getParams() { return kernelParams; };
    //void setParams(Vector params) { kernelParams = params; };
    virtual int getParamCount();
    virtual Vector getParams();
    virtual void setParams(Vector params);
    //Vector kernelParams = Vector(0);
    //int paramCount = 0;
  private:
    virtual double evalKernel(Matrix&, Matrix&, Vector&, int);
    virtual double evalDistKernel(double, Vector&, int);
  };
  */


  // Define class for radial basis function (RBF) covariance kernel
  class RBF // : public Kernel
  {
  public:
    // Constructors
    RBF() : kernelParams(Vector(1)) , paramCount(1) { kernelParams(0)=1.0; };
    //RBF() : Kernel(Vector(1),1) { kernelParams = Vector(1); kernelParams(0)=1.0; paramCount = 1; };
    void computeCov(Matrix & K, Matrix & D, Vector & params, int deriv);

    int getParamCount() { return paramCount; } ;
    Vector getParams() { return kernelParams; };
    void setParams(Vector params) { kernelParams = params; };

    //int getParamCount();
    //void setParams(Vector params);
    //Vector getParams();
    Vector kernelParams;
    int paramCount;
    
  private:
    double evalKernel(Matrix&, Matrix&, Vector&, int);
    double evalDistKernel(double, Vector&, int);
    //Vector params;
    //int paramCount;
  };
  
  // Define class for Gaussian processes
  class GaussianProcess : public minimize::GradientObj
  {    
  public:
    // Constructors
    GaussianProcess() : dimIn(1) { }
    GaussianProcess(int din) : dimIn(din) { }
    // Copy Constructor
    GaussianProcess(const GaussianProcess & m) :
      kernel(m.kernel),
      noiseLevel(m.noiseLevel),
      fixedNoise(m.fixedNoise),
      obsX(m.obsX) ,
      obsY(m.obsY)
      { N = static_cast<int>(obsX.rows()); }
      //{ std::cout << "\nCOPY\n"; N = static_cast<int>(obsX.rows()); }


    // Get and show methods
    double evalNLML(const Vector & p); //, Matrix & alpha);
    void evalDNLML(const Vector & p, Vector & g); //, Matrix & alpha);
    //Matrix getPredMean() { return predMean; }
    //Matrix getPredVar() { return predCov.diagonal() + noiseLevel*Eigen::VectorXd::Ones(predMean.size()); }
    //Matrix getSamples(int count=10);
    decltype(auto) getParams() { return kernel.getParams(); }
    double getNoise() { return noiseLevel; }
    
    // Set methods
    void setObs(Matrix & x, Matrix & y) { obsX = x; obsY = y; N = static_cast<int>(x.rows()); }
    //void setKernel(Kernel k) { kernel = k; }
    void setKernel(RBF k) { kernel = k; }
    void setPred(Matrix & px) { predX = px; }

    // NOISE
    void setNoise(double noise) { fixedNoise = true; noiseLevel = noise; }
    //void setNoise(double noise) { fixedNoise = true; noiseLevel = 0.0; }
    
    // Compute methods
    //void predict();
    void fitModel();

    // Define method for superclass "GradientObj" used by minimization algorithm
    void computeValueAndGradient(Vector X, double & val, Vector & D) { val = evalNLML(X); evalDNLML(X,D); };

    void computeDistMat();

      
  private:
    // Private member functions
    
    // Status variables
    int dimIn;
    int N = 0;

    // Kernel and covariance matrix
    //Kernel kernel;
    RBF kernel;    
    double noiseLevel = 0.0;
    bool fixedNoise = false;
    double jitter = 1e-7;
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
