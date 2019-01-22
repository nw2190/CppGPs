# CppGPs
Implementation of Numerical Gaussian Processes in C++

# Profiling
## Requirements
* [`valgrind`](http://valgrind.org/docs/manual/quick-start.html) - debugging/profiling tool suite
* [`perf`](https://en.wikipedia.org/wiki/Perf_(Linux)) - performance analyzing tool for Linux
* [`graphviz`](https://www.graphviz.org/) - open source graph visualization software
* [`gprof2dot`](https://github.com/jrfonseca/gprof2dot) - python script for converting profiler output to dot graphs

Profiling data is produced via the `callgrind` tool in the `valgrind` suite:
```
valgrind --tool=callgrind --trace-children=yes ./Run
```
__Note:__ This will take _much_ more time to run than the standard execution (e.g. 100x).


A graph visualization of the node-wise executation times in the program can then be created via:
```
perf record -g -- ./Run
perf script | c++filt | gprof2dot -s -n 5.0 -f perf | dot -Tpng -o output.png
```
[//]: # (COMMENT: perf script | c++filt | python /usr/lib/python3.7/site-packages/gprof2dot.py -f perf | dot -Tpng -o output.png)


_Note:_ The `-s` flag can also be removed from the `gprof2dot` call to show parameter types.  The `-n` flag is used to specify the percentage threshold for ommitting nodes.
