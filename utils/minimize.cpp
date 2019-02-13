#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Dense>
#include "minimize.h"

using Matrix = minimize::Matrix;
using Vector = minimize::Vector;


// Define interpolation procedure for minimization algorithm
double minimize::interpolate(double x2, double f2, double d2, double x3, double f3, double d3, double f0, double INT, double RHO)
{

  // choose subinterval
  // move point 3 to point 4
  double x4 = x3;
  double f4 = f3;
  double d4 = d3;

  //double tolerance = 1e-32;
  double tolerance = 1e-64;

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
double minimize::cubic_extrap(double x1, double x2, double f1, double f2, double d1, double d2, double EXT, double INT)
{
  // make cubic extrapolation
  double A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);
  double B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);

  double x3;
  //double tolerance = 1e-32;
  double tolerance = 1e-64;
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
    


//
//  ORIGINAL CODE BY CARL EDWARD RASMUSSEN
//  http://learning.eng.cam.ac.uk/carl/code/minimize/
//
//  % Minimize a differentiable multivariate function. 
//  %
//  % Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
//  %
//  % where the starting point is given by "X" (D by 1), and the function named in
//  % the string "f", must return a function value and a vector of partial
//  % derivatives of f wrt X, the "length" gives the length of the run: if it is
//  % positive, it gives the maximum number of line searches, if negative its
//  % absolute gives the maximum allowed number of function evaluations. You can
//  % (optionally) give "length" a second component, which will indicate the
//  % reduction in function value to be expected in the first line-search (defaults
//  % to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
//  %
//  % The function returns when either its length is up, or if no further progress
//  % can be made (ie, we are at a (local) minimum, or so close that due to
//  % numerical problems, we cannot get any closer). NOTE: If the function
//  % terminates within a few iterations, it could be an indication that the
//  % function values and derivatives are not consistent (ie, there may be a bug in
//  % the implementation of your "f" function). The function returns the found
//  % solution "X", a vector of function values "fX" indicating the progress made
//  % and "i" the number of iterations (line searches or function evaluations,
//  % depending on the sign of "length") used.
//  %
//  % The Polack-Ribiere flavour of conjugate gradients is used to compute search
//  % directions, and a line search using quadratic and cubic polynomial
//  % approximations and the Wolfe-Powell stopping criteria is used together with
//  % the slope ratio method for guessing initial step sizes. Additionally a bunch
//  % of checks are made to make sure that exploration is taking place and that
//  % extrapolation will not be unboundedly large.
//  %
//  % See also: checkgrad 
//  %
//  % Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).
//  
//  INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
//  EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
//  MAX = 20;                         % max 20 function evaluations per line search
//  RATIO = 10;                                       % maximum allowed slope ratio
//  SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
//  % Powell conditions. SIG is the maximum allowed absolute ratio between
//  % previous and new slopes (derivatives in the search direction), thus setting
//  % SIG to low (positive) values forces higher precision in the line-searches.
//  % RHO is the minimum allowed fraction of the expected (from the slope at the
//  % initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
//  % Tuning of SIG (depending on the nature of the function to be optimized) may
//  % speed up the minimization; it is probably not worth playing much with RHO.
//  
//  % The code falls naturally into 3 parts, after the initial line search is
//  % started in the direction of steepest descent. 1) we first enter a while loop
//  % which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
//  % have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
//  % enter the second loop which takes p2, p3 and p4 chooses the subinterval
//  % containing a (local) minimum, and interpolates it, unil an acceptable point
//  % is found (Wolfe-Powell conditions). Note, that points are always maintained
//  % in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
//  % conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
//  % was a problem in the previous line-search. Return the best value so far, if
//  % two consecutive line-searches fail, or whenever we run out of function
//  % evaluations or line-searches. During extrapolation, the "f" function may fail
//  % either with an error or returning Nan or Inf, and minimize should handle this
//  % gracefully.
//  


  
// Conjugate gradient minimization algorithm
void minimize::cg_minimize(Vector & X, minimize::GradientObj * target, Vector & D, int length, double SIG, double EXT, double INT, int MAX)
{
  // specify optimization hyperparameters
  //double RATIO = 10.0;
  double RATIO = 100.0;
  double RHO = SIG/2;

  // determine problem dimension
  int N = static_cast<int>(X.size());

  // initialize values
  int i = 0;
  bool ls_failed = false;
  Vector df0(N);
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

      // Display current parameter values
      //std::cout << " X  =  " << X.transpose().array().exp().matrix() << std::endl;
      
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
            {
              x3 = x3 * RATIO;
              std::cout << "\n[*] RATIO parameter enforced\n";
            }
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


          // DEBUGGING INFO TO SEE WHY OPTIMIZATION EXITS EARLY
          /*
          if ( ls_failed )
            {
              std::cout << "\n[*] Line Search Failed  ( i = " << i << " )\n";
              std::cout << "\n The following conditions failed:   [ SIG = " << SIG << " , RHO = " << RHO << " ]\n";
              if ( std::abs(d3) >= -SIG*d0 )
                {
                  double lhs = std::abs(d3);
                  double rhs = -SIG*d0;
                  std::cout << "abs(" << d3 << ")   <   -SIG * " << d0 << " [   i.e. " << lhs << " < " << rhs << " ]\n";
                }
              if ( f3 >= f0+x3*RHO*d0 )
                {
                  double lhs = f3;
                  double rhs = f0+x3*RHO*d0;
                  std::cout <<  f3 << "   <   " << f0 << " + " << x3 << " * RHO * " << d0 << "   [ i.e. " << lhs << " < " << rhs << " ]\n";
                }
            }
          if ( i > length )
            std::cout << "\n[*] Exceeded 'length' value\n";
          */
          
          // try steepest
          s = -df0;
          d0 = -s.transpose()*s;           
          x3 = 1/(1-d0);

          // this line search failed
          ls_failed = true;                   

        } // END COMPUTE NEW SEARCH DIRECTION

    }  // END MAIN LOOP
  
  //std::cout << "\nMinimized Function Value (???) :\n";
  //std::cout << f0 << std::endl;

};

