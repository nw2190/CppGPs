import numpy as np

sqrt = np.sqrt


def interpolate(x2, f2, d2, x3, f3, d3, f0, INT, RHO):
    # choose subinterval
    #if d3 > 0 or f3 >= f0+x3*RHO*d0:
    # move point 3 to point 4
    x4, f4, d4 = [x3, f3, d3]
    #else:
    #    print("\n\nTHIS WAS USED\n\n")
    #    # move point 3 to point 2
    #    x2, f2, d2 = [x3, f3, d3]

    #print(f4)
    if f4 > f0:
        tolerance = 1e-16
        denom = f4-f2-d2*(x4-x2)
        if abs(denom) < tolerance:
            x3 = (x2+x4)/2;
        else:
            # quadratic interpolation
            x3 = x2-(0.5*d2*np.power(x4-x2,2))/(denom);  
    else:
        # cubic interpolation
        A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                        
        B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);

        tolerance = 1e-16
        radical = B*B-A*d2*np.power(x4-x2,2)
        if (radical < 0) or (abs(A) < tolerance):
            x3 = (x2+x4)/2;
        else:
            x3 = x2+( sqrt(radical) - B)/A;
            
    # don't accept too close
    if x4-INT*(x4-x2) < x3:
        x3 = x4-INT*(x4-x2)
    if x2+INT*(x4-x2) > x3:
        x3 = x2+INT*(x4-x2) 
    #x3 = max(x3,x2+INT*(x4-x2));
    #x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));
    return x3


def cubic_extrap(x1, x2, f1, f2, d1, d2, EXT, INT):
    # make cubic extrapolation
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);

    tolerance = 1e-16
    radical = B*B-A*d1*(x2-x1)
    if radical < 0.0:
        x3 = x2*EXT
    elif B + sqrt(radical) < tolerance:
        x3 = x2*EXT
    else:
        x3 = x1-d1*np.power(x2-x1,2)/( B + sqrt(radical) );
        if (x3 < 0) or (x3 > x2*EXT):
            x3 = x2*EXT
        elif x3 < x2+INT*(x2-x1):
            x3 = x2+INT*(x2-x1)

    return x3


def cg_minimize(X, func, length, VERBOSE=False, SIG=0.1, EXT=3.0, INT=0.1):
    
    #INT = 0.1;    # don't reevaluate within 0.1 of the limit of the current bracket
    #EXT = 3.0;                  # extrapolate maximum 3 times the current step-size
    MAX = 20;                         # max 20 function evaluations per line search
    RATIO = 10;                                       # maximum allowed slope ratio
    RHO = SIG/2;

    # The code falls naturally into 3 parts, after the initial line search is
    # started in the direction of steepest descent.
    #
    # 1) We first enter a while loop which uses point 1 (p1) and point 2 (p2) to compute
    #    an extrapolation (p3), until we have extrapolated far enough (Wolfe-Powell conditions).
    #
    # 2) If necessary, enter the second loop which takes p2, p3 and p4 chooses the subinterval
    #    containing a (local) minimum, and interpolates it, unil an acceptable point is found
    #    (Wolfe-Powell conditions).
    #
    #    Note: points are always maintained in order p0 <= p1 <= p2 < p3 < p4.
    #
    # 3) Compute a new search direction using conjugate gradients (Polack-Ribiere flavour),
    #    or revert to steepest if there was a problem in the previous line-search.
    # 
    #    Return the best value so far, if two consecutive line-searches fail, or whenever we run out
    #    of function evaluations or line-searches.
    #
    #    During extrapolation, the "f" function may fail either with an error or returning Nan or Inf,
    #    and minimize should handle this gracefully.
    #

    # initialize values
    i = 0;
    ls_failed = 0;
    f0, df0 = func(X);
    fX = f0;

    # initial search direction (steepest) and slope
    # and the initial step is 1/(|s|+1)
    s = -df0; d0 = -np.matmul(np.transpose(s),s);
    x3 = 1/(1-d0);                                  

    ### MAIN LOOP
    request_break = False
    while i < length and not request_break:
        i = i + 1;

        # make a copy of current values
        X0, F0, dF0 = [X, f0, df0]

        # initialize iteration count
        M = MAX

        ### EXTRAPOLATE
        continue_extrap = True
        while continue_extrap:
            x2, f2, d2 = [0, f0, d0]

            M = M - 1; 
            f3, df3 = func(X+x3*s);

            # keep best values
            if f3 < F0:
                X0, F0, dF0 = [X+x3*s, f3, df3]

            # new slope                
            d3 = np.matmul(np.transpose(df3), s);                    

            # are we done extrapolating?
            if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0:           
                continue_extrap = False
                
            # move point 2 to point 1
            x1, f1, d1 = [x2, f2, d2]
            # move point 3 to point 2
            x2, f2, d2 = [x3, f3, d3]

            # cubic extrapolation
            x3 = cubic_extrap(x1, x2, f1, f2, d1, d2, EXT, INT)


        ### INTERPOLATE
        while (abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0:    

            x3 = interpolate(x2, f2, d2, x3, f3, d3, f0, INT, RHO)
            
            f3, df3 = func(X+x3*s);

            # keep best values
            if f3 < F0:
                X0, F0, dF0 = [X+x3*s, f3, df3]

            M = M - 1;          # count epochs?!
            d3 = np.matmul(np.transpose(df3),s);    # new slope
                                                    # end interpolation
                                                    
        ###
        ###  START COMPUTE NEW SEARCH DIRECTION
        ###
        if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:                           # if line search succeeded

            X = X+x3*s; f0 = f3; fX = np.transpose([np.transpose(fX), f0]);   # update variables            
            
            # Polack-Ribiere CG direction
            s = (np.matmul(np.transpose(df3),df3)-np.matmul(np.transpose(df0),df3))/(np.matmul(np.transpose(df0),df0))*s - df3;
            df0 = df3;                                              # swap derivatives
            d3 = d0; d0 = np.matmul(np.transpose(df0),s);
            if d0 > 0:                                              # new slope must be negative
                s = -df0; d0 = np.matmul(np.transpose(-s),s);       # otherwise use steepest direction

            # "realmin" = smallest positive normalized floating-point number in IEEE double precision format
            realmin = 2.2251e-308
            x3 = x3 * min(RATIO, d3/(d0-realmin));                  # slope ratio but max RATIO
            ls_failed = 0;                                          # this line search did not fail
            
        else:
            X = X0; f0 = F0; df0 = dF0;                             # restore best point so far
            
            if ls_failed or i > abs(length):                        # line search failed twice in a row
                request_break = True;                               # or we ran out of time, so we give up

            s = -df0; d0 = np.matmul(np.transpose(-s),s);           # try steepest
            x3 = 1/(1-d0);                     
            ls_failed = 1;                                          # this line search failed

    #print("\nIterations: {}\n".format(i))
    return X, fX, i


