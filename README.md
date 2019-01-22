# CppGPs
Implementation of Numerical Gaussian Processes in C++

# Profiling
Install [`valgrind`](http://valgrind.org/docs/manual/quick-start.html), [`graphviz`](https://github.com/jrfonseca/gprof2dot), and [`perf`](https://en.wikipedia.org/wiki/Perf_(Linux)).  Then run:

```
valgrind --tool=callgrind --trace-children=yes ./Run
perf record -g -- ./Run
perf script | c++filt | gprof2dot -s -n 5.0 -f perf | dot -Tpng -o output.png
```
[//]: # (COMMENT: perf script | c++filt | python /usr/lib/python3.7/site-packages/gprof2dot.py -f perf | dot -Tpng -o output.png)


Note: The `-s` flag can also be removed from the `gprof2dot` call to show parameter types.  The `-n` flag is used to specify the percentage threshold for ommitting nodes.
