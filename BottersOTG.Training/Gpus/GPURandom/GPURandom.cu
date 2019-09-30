#include "GPURandom.cuh"
#include "GPURandomState.generated.cu"

typedef unsigned int uint;

// Random number generators from: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html

// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.
__device__
uint TausStep(uint* z, int S1, int S2, int S3, uint M) {
	uint b = (((*z << S1) ^ *z) >> S2);
	return *z = (((*z & M) << S3) ^ b);
}

// A and C are constants
__device__
uint LCGStep(uint* z, uint A, uint C) {
  return *z = (A * (*z) + C);
}

__device__
uint HybridTaus(RandomState* state) {
	// Combined period is lcm(p1, p2, p3, p4) ~ 2^121
	return ( // Periods
		TausStep(&state->Taus1, 13, 19, 12, 4294967294UL) ^ // p1 = 2^31 - 1
		TausStep(&state->Taus2, 2, 25, 4, 4294967288UL) ^ // p2 = 2^30 - 1
		TausStep(&state->Taus3, 3, 11, 17, 4294967280UL) ^ // p3 = 2^28 - 1
		LCGStep(&state->Lcg, 1664525, 1013904223UL) // p4 = 2^32
	);
}

__device__
float randomFloat(RandomState* state, float maxValue) {
	const int Precision = 1e7;
	return (HybridTaus(state) % Precision) / (float)Precision * maxValue;
}

__device__
int randomInt(RandomState* state, int maxValue) {
	return HybridTaus(state) % maxValue;
}

__device__
int randomSharedInt(RandomState* state, int maxValue, int *sharedRandom) {
	if(threadIdx.x == 0) {
		sharedRandom[0] = randomInt(state, maxValue);
	}
	__syncthreads();
	return sharedRandom[0];
}

__device__
void _swapOrInc(int *a, int *b) {
	// ensure distinct & a < b
	if (*b < *a) {
		int temp = *a;
		*a = *b;
		*b = temp;
	} else {
		*b += 1;
	}
}

__device__
void Pick2Distinct(RandomState *randomState, int n, int *a, int *b) {
	// choose 2 values without replacement
	*a = randomInt(randomState, n);
	*b = randomInt(randomState, n - 1);
	if (*b >= *a) {
		*b += 1;
	}
}

__device__
void Pick2DistinctOrdered(RandomState *randomState, int offset, int n, int *a, int *b) {
	// choose 2 values without replacement and sort a < b
	*a = offset + randomInt(randomState, n);
	*b = offset + randomInt(randomState, n - 1);
	_swapOrInc(a, b);
}

__device__
void Pick3DistinctOrdered(RandomState *randomState, int offset, int n, int *a, int *b, int *c) {
	// choose 3 values without replacement and sort a < b < c
	*a = offset + randomInt(randomState, n);
	*b = offset + randomInt(randomState, n - 1);
	_swapOrInc(a, b);
	*c = offset + randomInt(randomState, n - 2);
	_swapOrInc(a, c);
	_swapOrInc(b, c);
}
