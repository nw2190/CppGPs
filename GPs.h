#ifndef _GPS_H
#define _GPS_H
#include <iostream>
#include <array>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include "./include/cppoptlib/meta.h"
//#include "./include/cppoptlib/problem.h"
#include "./include/cppoptlib/boundedproblem.h"
//#include "./include/cppoptlib/solver/bfgssolver.h"
#include "./include/cppoptlib/solver/lbfgsbsolver.h"

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
  class GaussianProcess
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
    double evalNLML(const Vector & p, Matrix & alpha);
    void evalDNLML(const Vector & p, Vector & g, Matrix & alpha);
    double getNLML() { return NLML; }
    Matrix getCov() { return K; }
    Matrix getPredMean() { return predMean; }
    Matrix getPredVar() { return predCov.diagonal() + noiseLevel*noiseLevel*Eigen::VectorXd::Ones(predMean.size()); }
    Matrix getSamples(int count=10);
    decltype(auto) getParams() { return kernelParams; }
    double getNoise() { return noiseLevel; }
    
    // Set methods
    void setObs(Matrix & x, Matrix & y) { obsX = x; obsY = y; N = static_cast<int>(x.rows()); }
    void setKernel(kernelptr k, Vector p) { kernelParams = p; paramCount = p.size(); kernel = std::move(k); }
    void setDistKernel(distkernelptr k, Vector p) { useDistKernel = true; kernelParams = p; paramCount = p.size(); distKernel = std::move(k); }
    void setPred(Matrix & px) { predX = px; }
    void setNoise(double noise) { fixedNoise = true; noiseLevel = noise; }
    
    // Compute methods
    void computeCov(Vector & p);
    void computeCov();
    void computeChol(double noise);
    void computeChol();
    void predict();
    void fitModel();
    
    // Define nested class for specifying minimization problem
    class fminProblem : public cppoptlib::BoundedProblem<double>  //  : public cppoptlib::Problem<double>
    {
    public:
      using Superclass = cppoptlib::BoundedProblem<double>;
      //fminProblem(std::unique_ptr<GaussianProcess> m) { model = std::move(m); }
      //fminProblem(std::unique_ptr<GaussianProcess> m, const Vector & lb, const Vector & ub) : Superclass(lb,ub) {model=std::move(m);}
      //std::unique_ptr<GaussianProcess> model;
      fminProblem(GaussianProcess * m, const Vector & lb, const Vector & ub) : Superclass(lb,ub), model(m)  { }
      double value(const Vector &x) { return (*model).evalNLML(x,_alpha); }
      void gradient(const Vector &x, Vector &g) { (*model).evalDNLML(x,g,_alpha); }      
      GaussianProcess * model;
    private:
      // Store alpha matrix for reuse in gradient calculation
      Matrix _alpha;
    };

    
    
  private:
    // Private member functions
    double evalKernel(Matrix&& x1, Matrix&& x2, int deriv=0) { return (*kernel)(x1, x2, kernelParams, deriv); }
    void computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p, int deriv);
    void computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2, Vector & p);
    void computeCrossCov(Matrix & M, Matrix & v1, Matrix & v2);

    // Status variables
    bool useDistKernel = false;
    int dimIn;
    int N = 0;

    // Kernel and covariance matrix
    double noiseLevel = 0.00001;
    bool fixedNoise = false;
    double jitter = 0.000001;
    kernelptr kernel;    
    distkernelptr distKernel;
    Vector kernelParams;
    int paramCount;
    Matrix K;
    Eigen::LLT<Matrix> cholesky;
    
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
