#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <chrono>
#include <random>
#include <boost/range/irange.hpp>
#include <Eigen/Dense>
#include "utils.h"

using Matrix = utils::Matrix;
using Vector = utils::Vector;

/*
template<typename T>
std::vector<T> utils::vlinspace(T a, T b, int n)
{
  std::vector<T> array;
  double step = (b-a) / (n-1);
  
  for (auto i : boost::irange(0,n))
    {
      array.push_back(a + i*step);
    }
  return array;
};
*/

/*
std::vector<float> utils::customFFT(std::vector<float> timevec, std::vector<std::complex<float> > freqvec)
{
  Eigen::FFT<float> fft;
  fft.fwd(freqvec,timevec);
  // manipulate freqvec
  fft.inv(timevec,freqvec);
  return timevec;
};
*/
