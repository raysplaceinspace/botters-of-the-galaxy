#pragma once

__device__ float findMaxF(const float* input, int length, float* localMax);

__device__ void sumKernelImpl(const int* input, int length, int& output, int* shared);
