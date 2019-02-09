import numpy as np

import sympy as sym
from sympy import sin as Sin
from sympy import cos as Cos
from sympy import exp as Exp



from cg_minimize import cg_minimize

# Define true solution
x, y = sym.symbols('x, y')


# Define kernel for u(x)
#func = ( (x - 0.2)**2 + (y + 0.4)**2 ) / (1 + x**2 + 0.1*y**4)
func = 100.0 * ( (x - 0.2)**2 + (y + 0.4)**2 ) / (1 + x**2 + 0.1*y**4)
func = sym.simplify(func)
func_x = sym.diff(func, x)
func_x = sym.simplify(func_x)
func_y = sym.diff(func, y) 
func_y = sym.simplify(func_y)

def eval_func(x_val, y_val):
    return func.subs(x,x_val).subs(y,y_val).evalf()

def eval_func_x(x_val, y_val):
    return func_x.subs(x,x_val).subs(y,y_val).evalf()

def eval_func_y(x_val, y_val):
    return func_y.subs(x,x_val).subs(y,y_val).evalf()


# Define test function
def func_and_deriv(X):
    x_val = X[0]
    y_val = X[1]
    func_val = eval_func(x_val, y_val)
    deriv_x_val = eval_func_x(x_val, y_val)
    deriv_y_val = eval_func_y(x_val, y_val)

    dtype = np.float64
    func_val = np.array(func_val, dtype=dtype)
    deriv_vals = np.array([deriv_x_val, deriv_y_val], dtype=dtype)

    return [func_val, deriv_vals]


# Test minimization function
def main():

    if False:
        func_code = sym.printing.ccode(func)
        func_x_code = sym.printing.ccode(func_x)
        func_y_code = sym.printing.ccode(func_y)
        print(" ")
        print(func_code)
        print(" ")
        print(func_x_code)
        print(" ")
        print(func_y_code)
        print(" ")
    
    X0 = [0.0, 0.0]
    f = func_and_deriv
    length = 1000
    INT = 0.01
    output = cg_minimize(X0, f, length, INT=INT, VERBOSE=False)

    arg = output[0]
    x_val = arg[0]
    y_val = arg[1]

    print(x_val)
    print(y_val)
    print(f([x_val,y_val]))
    

# Run main() function when called directly
if __name__ == '__main__':
    main()
