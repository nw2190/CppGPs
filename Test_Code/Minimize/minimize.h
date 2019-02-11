#ifndef _MINIMIZE_H
#define _MINIMIZE_H
#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>
#include <Eigen/Dense>

// Declare namespace for utils
namespace minimize {

  // Define aliases with using declarations
  using Matrix = Eigen::MatrixXd;
  using Vector = Eigen::VectorXd;
  using Eigen::VectorXcd;
  //using VectorXld = Eigen::Matrix<long double, Eigen::Dynamic, 1>;

  // Define aliases for target minimization function
  //using minimizefn = double (*)(Vector, Vector&);
  //using minimizeptr = std::unique_ptr<minimizefn>;    

  class GradientObj
  {
  public:
    GradientObj() { };
    virtual void computeValueAndGradient(Vector,double&,Vector&) { };
  };
      
  
  // Define interpolation procedure for minimization algorithm
  double interpolate(double x2, double f2, double d2, double x3, double f3, double d3, double f0, double INT, double RHO);

  // Define cubic extrapolation routine for minimization algorithm
  double cubic_extrap(double x1, double x2, double f1, double f2, double d1, double d2, double EXT, double INT);

  
  // Conjugate gradient minimization algorithm
  //void cg_minimize(Vector & X, minimizeptr func, Vector & D, int length, double SIG=0.1, double EXT=3.0, double INT=0.01)
  //void cg_minimize(Vector & X, GradientObj * target, Vector & D, int length, double SIG=0.1, double EXT=3.0, double INT=0.01);
  void cg_minimize(Vector & X, GradientObj * target, Vector & D, int length, double SIG=0.1, double EXT=3.0, double INT=0.01, int MAX=20);

  
};
#endif
