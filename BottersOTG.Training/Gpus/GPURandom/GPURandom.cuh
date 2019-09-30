#pragma once

#include "GPURandomState.generated.cu"

__device__ unsigned int HybridTaus(RandomState* state);

__device__ float randomFloat(RandomState* state, float maxValue); 

__device__ int randomInt(RandomState* state, int maxValue);

__device__ int randomSharedInt(RandomState* state, int maxValue, int *sharedRandom);

__device__ void Pick2Distinct(RandomState *randomState, int n, int *a, int *b);

__device__ void Pick2DistinctOrdered(RandomState *randomState, int offset, int n, int *a, int *b);

__device__ void Pick3DistinctOrdered(RandomState *randomState, int offset, int n, int *a, int *b, int *c);