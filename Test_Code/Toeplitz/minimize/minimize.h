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
  double interpolate(double x2, double f2, double d2, double x3, double f3, double d3, double f0, double INT, double RHO)
  {

    // choose subinterval
    // move point 3 to point 4
    double x4 = x3;
    double f4 = f3;
    double d4 = d3;
    
    double tolerance = 1e-32;
    
    if ( f4 > f0 )
      {
        double denom = f4-f2-d2*(x4-x2);
        if ( std::abs(denom) < tolerance )
          // bisect
          x3 = (x2+x4)/2;
        else
          // quadratic interpolation
          x3 = x2-(0.5*d2*std::pow(x4-x2,2))/(denom);
      }
    else
      {
        // cubic interpolation
        double A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                        
        double B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
        double radical = B*B-A*d2*std::pow(x4-x2,2);

        if ( ( radical < 0 ) || ( std::abs(A) < tolerance ) )
          x3 = (x2+x4)/2;
        else
          x3 = x2+( std::sqrt(radical) - B)/A;
      }
    
    // don't accept too close
    if ( x4-INT*(x4-x2) < x3 )
      x3 = x4-INT*(x4-x2);
    
    if ( x2+INT*(x4-x2) > x3 )
      x3 = x2+INT*(x4-x2);

    return x3;
  };



  // Define cubic extrapolation routine for minimization algorithm
  double cubic_extrap(double x1, double x2, double f1, double f2, double d1, double d2, double EXT, double INT)
  {
    // make cubic extrapolation
    double A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);
    double B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);

    double x3;
    double tolerance = 1e-32;
    double radical = B*B-A*d1*(x2-x1);
    
    if ( radical < 0.0 )
      x3 = x2*EXT;
    else if ( B + std::sqrt(radical) < tolerance )
      x3 = x2*EXT;
    else
      {
        x3 = x1-d1*std::pow(x2-x1,2)/( B + std::sqrt(radical) );
        
        if ( ( x3 < 0 ) || ( x3 > x2*EXT ) )
          x3 = x2*EXT;
        else if ( x3 < x2+INT*(x2-x1) )
          x3 = x2+INT*(x2-x1);
      }

    return x3;
  };
    

  
  // Conjugate gradient minimization algorithm
  //void cg_minimize(Vector & X, minimizeptr func, Vector & D, int length, double SIG=0.1, double EXT=3.0, double INT=0.01)
  void cg_minimize(Vector & X, GradientObj * target, Vector & D, int length, double SIG=0.1, double EXT=3.0, double INT=0.01)
  {
    // specify optimization hyperparameters
    int MAX = 20;
    double RATIO = 10.0;
    double RHO = SIG/2;

    // determine problem dimension
    int N = static_cast<int>(X.size());
    
    // initialize values
    int i = 0;
    bool ls_failed = false;
    Vector df0(N);
    //double f0 = (*func)(X, df0);
    double f0;
    (*target).computeValueAndGradient(X, f0, df0);
    
    // initial search direction (steepest) and slope 
    Vector s = -df0;
    double d0 = -s.transpose()*s;

    // initial step is 1/(|s|+1)
    double x3 = 1/(1-d0);     

    // declare placeholders for storing optimal values
    Vector X0(N);
    double F0;
    Vector dF0(N);

    // declare variables in main loop
    int M;
    bool continue_extrap;
    double x1, f1, d1, x2, f2, d2, f3, d3;
    Vector df3(N);

    // "realmin" = smallest positive normalized floating-point number in IEEE double precision format
    double realmin = 2.2251e-308;
    
    // MAIN LOOP
    bool request_break = false;
    while ( ( i < length ) && ( !request_break ) )
      {
        i++;

        // make a copy of current values
        X0 = X;
        F0 = f0;
        dF0 = df0;

        // initialize iteration count
        M = MAX;

        // EXTRAPOLATE
        continue_extrap = true;
        while ( continue_extrap )
          {
            x2 = 0.0;
            f2 = f0;
            d2 = d0;
            M = M - 1;
            //f3  = (*func)(X+x3*s, df3);
            (*target).computeValueAndGradient(X+x3*s, f3, df3);

            // keep best values
            if ( f3 < F0 )
              {
                X0 = X+x3*s;
                F0 = f3;
                dF0 = df3;
              }

            // new slope                
            d3 = df3.transpose()*s;                    

            // are we done extrapolating?
            if ( ( ( d3 > SIG*d0 ) || ( f3 > f0+x3*RHO*d0 ) ) || ( M == 0 ) )
                continue_extrap = false;
                
            // move point 2 to point 1
            x1 = x2;
            f1 = f2;
            d1 = d2;
            // move point 3 to point 2
            x2 = x3;
            f2 = f3;
            d2 = d3;

            // cubic extrapolation
            x3 = cubic_extrap(x1, x2, f1, f2, d1, d2, EXT, INT);
            
          } // END EXTRAPOLATE

        
        // INTERPOLATE
        while ( ( ( std::abs(d3) > -SIG*d0 ) || ( f3 > f0+x3*RHO*d0) )  &&  ( M > 0 ) )
          {

            x3 = interpolate(x2, f2, d2, x3, f3, d3, f0, INT, RHO);
            
            //f3 = (*func)(X+x3*s, df3);
            (*target).computeValueAndGradient(X+x3*s, f3, df3);

            // keep best values
            if ( f3 < F0 )
              {
                X0 = X+x3*s;
                F0 = f3;
                dF0 = df3;
              }

            // decrement line-search count
            M = M - 1;
            
            // new slope            
            d3 = df3.transpose()*s;
            
          } // END INTERPOLATE



        //  START COMPUTE NEW SEARCH DIRECTION
        if ( ( std::abs(d3) < -SIG*d0 ) && ( f3 < f0+x3*RHO*d0 ) )            
          {
            // if line search succeeded
            // update variables            
            X = X+x3*s;
            f0 = f3;
            
            // Polack-Ribiere CG direction
            s = ( (df3.transpose()*df3 - df0.transpose()*df3)(0) / (df0.transpose()*df0)(0) )*s - df3;

            // swap derivatives
            df0 = df3;
            d3 = d0;
            d0 = df0.transpose()*s;

            // new slope must be negative
            if ( d0 > 0 )
              {
                // otherwise use steepest direction
                s = -df0;
                d0 = -s.transpose()*s;
              }

            // slope ratio but max RATIO
            if ( RATIO <  d3/(d0-realmin) )
                x3 = x3 * RATIO;
            else
                x3 = x3 * d3/(d0-realmin);

            // this line search did not fail
            ls_failed = false;                                          
          }
        
        else
          {
            // restore best point so far
            X = X0;
            f0 = F0;
            df0 = dF0;                             

            // line search failed twice in a row
            // or we ran out of time, so we give up
            if ( ( ls_failed ) || ( i > length ) )        
                request_break = true;                               

            // try steepest
            s = -df0;
            d0 = -s.transpose()*s;           
            x3 = 1/(1-d0);
            
            // this line search failed
            ls_failed = true;                   
        
          } // END COMPUTE NEW SEARCH DIRECTION


      }  // END MAIN LOOP
    
  };
  

  
};
#endif