//#define EIGEN_USE_MKL_ALL
#ifndef _GPS_H
#define _GPS_H
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
//#include "./utils/minimize.h"

// Include CppOptLib files
#include "./include/cppoptlib/meta.h"
#include "./include/cppoptlib/boundedproblem.h"
#include "./include/cppoptlib/solver/lbfgsbsolver.h"


// Declare namespace for Gaussian process definitions
namespace GP {

  // Define PI using arctan function
  static const double PI = std::atan(1)*4;

  // Define aliases with using declarations
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;

  // Define function for retrieving time from chrono
  float getTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);
  using time = std::chrono::high_resolution_clock::time_point;
  using std::chrono::high_resolution_clock;
  
  // Define function for sampling uniform distribution on interval
  Matrix sampleUnif(double a=0.0, double b=1.0, int N=1);
  Vector sampleUnifVector(Vector lbs, Vector ubs);

  // Define utility functions for computing distance matrices
  void pdist(Matrix & Dv, Matrix & X1, Matrix & X2);
  void squareForm(Matrix & D, Matrix & Dv, int n, double diagVal=0.0);
  
  // Define abstract base class for covariance kernels
  class Kernel 
  {    
  public:
    // Constructors
    Kernel(Vector p, int c) : kernelParams(p) , paramCount(c) { };
    virtual ~Kernel() = default;
    virtual std::vector<Matrix> computeCov(Matrix & K, Matrix & D, Vector & params, double jitter=0.0, bool evalGrad=false) = 0;
    virtual void computeCrossCov(Matrix & K, Matrix & X1, Matrix & X2, Vector & params) = 0;
    void setNoise(double noise) { noiseLevel = noise; }
    int getParamCount() { return paramCount; } ;
    Vector getParams() { return kernelParams; };
    void setParams(Vector params) { kernelParams = params; };
  protected:
    Vector kernelParams;
    int paramCount;
    double noiseLevel;
    virtual double evalKernel(Matrix&, Matrix&, Vector&, int) = 0;
    virtual double evalDistKernel(double, Vector&, int) = 0;
  };


  // Define class for radial basis function (RBF) covariance kernel
  class RBF : public Kernel
  {
  public:
    // Constructors
    RBF() : Kernel(Vector(1), 1) { kernelParams(0)=1.0; };
    std::vector<Matrix> computeCov(Matrix & K, Matrix & D, Vector & params, double jitter=0.0, bool evalGrad=false);
    void computeCrossCov(Matrix & K, Matrix & X1, Matrix & X2, Vector & params);
  private:
    double evalKernel(Matrix&, Matrix&, Vector&, int);
    double evalDistKernel(double, Vector&, int);
    Matrix Kv;
    Matrix dK_i;
    Matrix dK_iv;
  };



  
  // Define class for Gaussian processes
  class GaussianProcess : public cppoptlib::BoundedProblem<double>
  {    
  public:

    using Superclass = cppoptlib::BoundedProblem<double>;
    
    // Constructors  [ Initialize lowerBounds / upperBounds of CppOptLib superclass to zero ]
    GaussianProcess() : Superclass(Vector(0), Vector(0)) { }

    // Define CppOptLib methods
    double value(const Vector &p) { return evalNLML(p, cppOptLibgrad, true); }
    void gradient(const Vector &p, Vector &g) { g = cppOptLibgrad; }
    Vector cppOptLibgrad;

    // Set methods
    void setObs(Matrix & x, Matrix & y) { obsX = x; obsY = y; N = static_cast<int>(x.rows()); }
    void setKernel(Kernel & k) { kernel = &k; }
    void setPred(Matrix & px) { predX = px; }
    void setNoise(double noise) { fixedNoise = true; noiseLevel = noise; }
    void setBounds(Vector & lbs, Vector & ubs) { lowerBounds = lbs; upperBounds = ubs; fixedBounds=true; }
    
    // Compute methods
    void fitModel();
    void predict();
    double computeNLML(const Vector & p, double noise);
    double computeNLML(const Vector & p);
    double computeNLML();
    
    // Get methods    
    Matrix getPredMean() { return predMean; }
    Matrix getPredVar() { return predCov.diagonal() + noiseLevel*Eigen::VectorXd::Ones(predMean.size()); }
    Matrix getSamples(int count=10);
    Vector getParams() { return (*kernel).getParams(); }
    double getNoise() { return noiseLevel; }
    

    // Define method for superclass "GradientObj" used by minimization algorithm
    void computeValueAndGradient(Vector X, double & val, Vector & D) { val = evalNLML(X,D,true); };

      
  private:
    
    // Private member functions
    double evalNLML(const Vector & p); 
    double evalNLML(const Vector & p, Vector & g, bool evalGrad=false);
    void computeDistMat();
    
    // Status variables  ( still needed ? )
    int N = 0;

    // Kernel and covariance matrix
    Kernel * kernel;
    double noiseLevel = 0.0;
    bool fixedNoise = false;
    double jitter = 1e-7;
    Matrix K;
    Eigen::LLT<Matrix> cholesky;

    Vector lowerBounds;
    Vector upperBounds;
    bool fixedBounds = false;
    void parseBounds(Vector & lbs, Vector & ubs, int augParamCount);
      
    // Store squared distance matrix and alpha for NLML/DNLML calculations
    Matrix distMatrix;    
    Matrix _alpha;
    
    // Observation data
    Matrix obsX; 
    Matrix obsY; 
    
    // Prediction data
    Matrix predX;
    Matrix predMean;
    Matrix predCov;
    double NLML = 0.0;

    int paramCount;
    int augParamCount;
    Matrix term;
    //std::vector<Matrix> gradList;  // Need to find a way to avoid creating new gradList each time...

    // DEFINE TIMER VARIABLES
    /*
    double time_computecov = 0.0;
    double time_cholesky_llt = 0.0;
    double time_NLML = 0.0;
    double time_term = 0.0;
    double time_grad = 0.0;
    double time_paramsearch = 0.0;
    double time_minimize = 0.0;
    double time_final_chol = 0.0;
    */
  };

  // Define linspace function for generating
  // equally spaced points on an interval
  template <typename T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> linspace(T a, T b, int N)
  { return Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(N, a, b); }

    
  
};


#endif
