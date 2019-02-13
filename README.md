# CppGPs
Implementation of Numerical Gaussian Processes in C++

## Dependencies
* [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) - High-level C++ library for linear algebra, matrix operations, and solvers
* [GCC](https://gcc.gnu.org/) - GNU compiler collection; more specifically the GCC C++ compiler is recommended
* [NumPy](http://www.numpy.org/)* - Scientific computing package for Python
* [MatPlotLib](https://matplotlib.org/)* - Python plotting library
* [SciKit Learn](https://scikit-learn.org/stable/)* - Data analysis library for Python



\* Optional dependencies



## Regression

### CppGP Implementation
The `main.cpp` file provides an example for how to use the CppGP code for Gaussian process regression.  The `targetFunc` function is used to define artificial data for the regression task on the input values `x`.  The level of noise in the training observations can also be adjusted via the `noiseLevel` parameter.  The artificial observation and corresponding predictions/samples are saved in the `observations.csv` and `predictions.csv`/`samples.csv` files, respectively.


### Comparison with SciKit Learn Implementation

The results of the CppGP code and SciKit Learn `GaussianProcessRegressor` class can be compared using the `SciKit_Learn_Comparison.py` Python script.  This code provides the estimated kernel/noise parameters and negative log marginal likelihood (NLML) calculations in addition to plots of the CppGP and SciKit Learn results.




## Profiling

### Requirements
* [`valgrind`](http://valgrind.org/docs/manual/quick-start.html) - debugging/profiling tool suite
* [`perf`](https://en.wikipedia.org/wiki/Perf_(Linux)) - performance analyzing tool for Linux
* [`graphviz`](https://www.graphviz.org/) - open source graph visualization software
* [`gprof2dot`](https://github.com/jrfonseca/gprof2dot) - python script for converting profiler output to dot graphs

Profiling data is produced via the `callgrind` tool in the `valgrind` suite:
```
valgrind --tool=callgrind --trace-children=yes ./Run
```
__Note:__ This will take _much_ more time to run than the standard execution time (e.g. 100x).


A graph visualization of the node-wise executation times in the program can then be created via:
```
perf record -g -- ./Run
perf script | c++filt | gprof2dot -s -n 5.0 -f perf | dot -Tpng -o output.png
```
[//]: # (COMMENT: perf script | c++filt | python /usr/lib/python3.7/site-packages/gprof2dot.py -f perf | dot -Tpng -o output.png)


__Note:__ The `-s` flag can also be removed from the `gprof2dot` call to show parameter types.  The `-n` flag is used to specify the percentage threshold for ommitting nodes.
