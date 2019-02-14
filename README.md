# CppGPs
Implementation of Numerical Gaussian Processes in C++

## Dependencies
* [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) - High-level C++ library for linear algebra, matrix operations, and solvers
* [GCC](https://gcc.gnu.org/) - GNU compiler collection; more specifically the GCC C++ compiler is recommended
* [CppOptLib](https://github.com/PatWie/CppNumericalSolvers) - A header-only optimization library with a C++ L-BFGS implementation

### Optional Dependencies for Plotting / SciKit Learn Comparison
* [NumPy](http://www.numpy.org/) - Scientific computing package for Python
* [csv](https://docs.python.org/3/library/csv.html) - Python module for working with comma separated value (CSV) files
* [MatPlotLib](https://matplotlib.org/) - Python plotting library
* [SciKit Learn](https://scikit-learn.org/stable/) - Data analysis library for Python



## Gaussian Process Regression

### Compiling and Running the Code
The `main.cpp` file provides an example use of the CppGP code for Gaussian process regression.  After specifying the correct path to the Eigen header files by editing the `EIGENPATH` variable in the `makefile`, the example code can be compiled and run as follows:
```console
user@host $ make all
g++ -c -Wall  -std=c++17 -I/usr/include/eigen3 -g -march=native -fopenmp -O3 main.cpp -o main.o
g++ -c -Wall  -std=c++17 -I/usr/include/eigen3 -g -march=native -fopenmp -O3 GPs.cpp -o GPs.o
g++ -std=c++17 -I/usr/include/eigen3 -g -march=native -fopenmp -O3 -o Run main.cpp GPs.cpp

user@host $ ./Run

Computation Time: 3.47298 s

Optimized Hyperparameters:
0.0345955  (Noise = 0.33991)

NLML:  1039.64

```

### Defining the Target Function and Training Data
The `targetFunc` function defined at the beginning of the `main.cpp` file is used to generate artificial training data for the regression task:
```cpp
// Specify the target function for Gaussian process regression
double targetFunc(double x)
{
  double oscillation = 30.0;
  return std::sin(oscillation*(x-0.1))*(0.5-(x-0.1))*15.0;
}
```
The training data consists of a collection of input points `X` along with an associated collection of target values `y`.  This data should be formatted so that `y(i) = targetFunc(X.row(i))` (with an optional additive noise term).  A simple one-dimensional problem setup can be defined as follows:
```cpp
int obsCount = 1000;
Matrix X = sampleUnif(0.0, 1.0, obsCount);
Matrix y;  y.resize(obsCount, 1);
```
Noise can be added to the training target data `y` to better assess the fit of the model's predictive variance.  The level of noise in the training data can be adjusted via the `noiseLevel` parameter and used to define the target data via:
```cpp
auto noiseLevel = 1.0;
auto noise = Eigen::VectorXd::Random(obsCount) * noiseLevel;
y = X.unaryExpr(std::ptr_fun(targetFunc)) + noise;
```

### Specifying and Fitting the Gaussian Process Model

```cpp
// Initialize Gaussian process regression model
GaussianProcess model;

// Specify training observations for GP model
model.setObs(X,y);

// Initialize RBF kernel and assign it to the model
RBF kernel;
model.setKernel(kernel);

// Specify hyperparameter bounds
Vector lbs(1);  lbs <<  0.01;
Vector ubs(1);  ubs <<  100.0;
model.setBounds(lbs, ubs);

// Fit model to the training data
model.fitModel();  
```

### Posterior Predictions and Sample Paths
```cpp
// Define test mesh for GP model predictions
int predCount = 100;
auto testMesh = linspace(0.0, 1.0, predCount);
model.setPred(testMesh);

// Compute predicted means and variances for the test points
model.predict();
Matrix pmean = model.getPredMean();
Matrix pvar = model.getPredVar();
Matrix pstd = (pvar.array().sqrt()).matrix();

// Get sample paths from the posterior distribution of the model
int sampleCount = 100;
Matrix samples = model.getSamples(sampleCount);
```

### Plotting Results of the Trained Gaussian Process Model
The artificial observation data and corresponding predictions/samples are saved in the `observations.csv` and `predictions.csv`/`samples.csv` files, respectively.  The trained model results can be plotted using the provided Python script `Plot.py`.


### Comparison with SciKit Learn Implementation

The results of the CppGP code and SciKit Learn `GaussianProcessRegressor` class can be compared using the `SciKit_Learn_Comparison.py` Python script.  This code provides the estimated kernel/noise parameters and negative log marginal likelihood (NLML) calculations in addition to plots of the CppGP and SciKit Learn results.




## Profiling the CppGPs Implementation

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
