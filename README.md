# CppGPs
Implementation of Numerical Gaussian Processes in C++

# Profiling
Install [`graphviz`](https://github.com/jrfonseca/gprof2dot) and [`perf`](https://en.wikipedia.org/wiki/Perf_(Linux)).  Then run:

```
perf record -g -- ./Run
perf script | c++filt | gprof2dot -f perf | dot -Tpng -o output.png
```
[//]: # (COMMENT: perf script | c++filt | python /usr/lib/python3.7/site-packages/gprof2dot.py -f perf | dot -Tpng -o output.png)
