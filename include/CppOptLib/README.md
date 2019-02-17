# CppOptimizationLibrary

The CppOptimizationLibrary can be downloaded from the GitHub repository [PatWie/CppNumericalSolvers](https://github.com/PatWie/CppNumericalSolvers).  Alternatively, the repository can be cloned via:

```console
user@host $ git clone git@github.com:PatWie/CppNumericalSolvers.git
```

The CppOptimizationLibrary's `cppoptlib` directory should then be moved to the CppGPs `include/` directory:

```console
user@host $ cp -r CppNumericalSolvers/include/cppoptlib ./include/
```

## Update L-BFGS Code

The original implementation of the L-BFGS algorithm provided by CppOptimizationLibrary needs to be slightly modified for compatibility with CppGPs.  In particular, the provided `lbfgsbsolver.h` file should be copied into the `cppoptlib/solvers/` directory and the `morethuente.h` file should be copied into the `cppoptlib/linesearch/` (overwriting the orginal files):

```console
user@host $ cp ./include/lbfgsbsolver.h ./include/cppoptlib/solver/
user@host $ cp ./include/morethuente.h ./include/cppoptlib/linesearch/
```
