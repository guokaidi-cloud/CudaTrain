#ifndef TIMER_HPP
#define TIMER_HPP

#include "cuda_runtime.h"
#include "cuda_utils.h"
#include <chrono>
#include <ratio>
#include <string>

class Timer {
public:
  using s = std::ratio<1, 1>;
  using ms = std::ratio<1, 1000>;
  using us = std::ratio<1, 1000000>;
  using ns = std::ratio<1, 1000000000>;

public:
  Timer();
  ~Timer();

public:
  void start_cpu();
  void start_gpu();
  void stop_cpu();
  void stop_gpu();

  template <typename span> void duration_cpu(std::string msg);

  void duration_gpu(std::string msg);

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> cstart_;
  std::chrono::time_point<std::chrono::high_resolution_clock> cstop_;
  cudaEvent_t gstart_;
  cudaEvent_t gstop_;
  float time_elasped_;
};

template <typename span> void Timer::duration_cpu(std::string msg) {
  std::string str;

  if (std::is_same<span, s>::value) {
    str = "s";
  } else if (std::is_same<span, ms>::value) {
    str = "ms";
  } else if (std::is_same<span, us>::value) {
    str = "us";
  } else if (std::is_same<span, ns>::value) {
    str = "ns";
  }

  std::chrono::duration<double, span> time = cstop_ - cstart_;
  LOG("%-40s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
}

#endif // TIMER_HPP
