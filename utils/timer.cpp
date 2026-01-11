#include "timer.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda_utils.h"
#include <chrono>
#include <iostream>
#include <memory>

Timer::Timer() {
  time_elasped_ = 0;
  cstart_ = std::chrono::high_resolution_clock::now();
  cstop_ = std::chrono::high_resolution_clock::now();
  cudaEventCreate(&gstart_);
  cudaEventCreate(&gstop_);
}

Timer::~Timer() {
  cudaEventDestroy(gstart_);
  cudaEventDestroy(gstop_);
}

void Timer::start_gpu() { cudaEventRecord(gstart_, 0); }

void Timer::stop_gpu() { cudaEventRecord(gstop_, 0); }

void Timer::start_cpu() { cstart_ = std::chrono::high_resolution_clock::now(); }

void Timer::stop_cpu() { cstop_ = std::chrono::high_resolution_clock::now(); }

void Timer::duration_gpu(std::string msg) {
  CUDA_CHECK(cudaEventSynchronize(gstart_));
  CUDA_CHECK(cudaEventSynchronize(gstop_));
  cudaEventElapsedTime(&time_elasped_, gstart_, gstop_);

  LOG("%-60s uses %.6lf ms", msg.c_str(), time_elasped_);
}
