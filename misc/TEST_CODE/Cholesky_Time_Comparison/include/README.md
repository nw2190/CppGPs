# CppOptimizationLibrary

The CppOptimizationLibrary can be downloaded from the GitHub repository [PatWie/CppNumericalSolvers](https://github.com/PatWie/CppNumericalSolvers).  Alternatively, the repository can be cloned via:

```console
user@host $ git clone git@github.com:PatWie/CppNumericalSolvers.git
```

The `cppoptlib` directory should then be placed here (i.e. `include/cppoptlib`) before compiling the CppGPs code.


## Update L-BFGS Code

The original implementation of the L-BFGS algorithm provided by CppOptimizationLibrary must be slightly modified for use with CppGPs.  In particular, the provided `lbfgsbsolver.h` file in this directory should be copied into the `cppoptlib/solvers/` directory and the `morethuente.h` file in this directory should be copied into the `cppoptlib/linesearch/` (overwriting the orginal files).
