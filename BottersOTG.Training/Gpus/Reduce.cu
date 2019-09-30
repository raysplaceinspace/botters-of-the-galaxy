#pragma once

#include "ThreadID.cuh"

#define __kernel extern "C" __global__

__device__
float findLocalMaxF(
	float* localMax) {

	__syncthreads();

	int self = threadIdx.x;
	for (int localStep = 1; localStep < blockDim.x; localStep <<= 1) {
		int other = self ^ localStep; // butterfly reduction
		localMax[self] = max(localMax[self], localMax[other]);
		__syncthreads(); 
	}

	return localMax[0];
}

__device__
float findMaxF(
	const float* input,
	int length,
	float* localMax) {

	float maxValue = 0.0f;
	for (int routeId = threadIdx.x; routeId < length; routeId += blockDim.x) {
		maxValue = max(maxValue, input[routeId]);
	}
	localMax[threadIdx.x] = maxValue;

	return findLocalMaxF(localMax);
}



/**
 * @brief Modifies the vector pointed by v so that the first element
 * holds the sum of the vector values. Assumes that v points to a
 * vector of size equal to blockDim.x
 */
__device__
void reduceSumOnSharedMemory(int* v) {

	for(unsigned reducedVectorSize = blockDim.x >> 1; reducedVectorSize > 0; reducedVectorSize >>= 1) {
		if(threadIdx.x < reducedVectorSize) {
			v[threadIdx.x] += v[threadIdx.x + reducedVectorSize];
		}
		__syncthreads();
		// Invariant: after every iteration the sum of the elements in
		// the range [0, reducedVectorSize) is equal to the sum of
		// elements in the original vector
	}
}

/**
 * @brief When called on all threads, the input vector with the given
 * length gets sliced on the shared memory pointed by shared. Shared
 * size must be equal to blockDim.x. In case the total shared memory
 * (summed on every block in the grid) is larger than length, the
 * shared memory is padded with 0 values. In case it is smaller, then
 * values will be summed on the shared memory, modulo gridDim.x *
 * blockDim.x
 */
__device__
void sliceSumOnSharedMemory(const int* input, int length, int* shared) {
	int sum = 0;
	for(unsigned i = get_global_id(0); i < length; i += get_global_size(0)) {
		sum += input[i];
	}

	shared[threadIdx.x] = sum;
	__syncthreads();
}

__device__
void sumKernelImpl(const int* input, int length, int& output, int* shared) {
	sliceSumOnSharedMemory(input, length, shared);
	
	reduceSumOnSharedMemory(shared);
	
	// every block contains the sum of elements within the block
	// itself, stored in localSum[0]
	if(threadIdx.x == 0) {
		atomicAdd(&output, shared[0]);
	}
}

/**
 * @brief Sets output to the input vector of the given length. Output
 * must be initialized to 0 (or whatever value you want to start
 * summing from). Requires an amount of shared memory equal to
 * threadIdx.x * sizeof(int)
 */
__kernel
void sumToOutputKernel(const int* input, int length, int* output) {
	extern __shared__ int rgSumKernelWorkspace[];
	int* __restrict__ shared = rgSumKernelWorkspace;
	sumKernelImpl(input, length, *output, shared);
}

/**
 * @brief Sets a single variable to the given value on the GPU
 */
__kernel
void setIntKernel(int* var, int value) {
	if(get_global_id(0) == 0) {
		*var = value;
	}
}
