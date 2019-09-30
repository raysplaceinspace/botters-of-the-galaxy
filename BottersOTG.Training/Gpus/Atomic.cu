#include "Atomic.cuh"

__device__ bool tryAtomicStore(volatile int* ref, int oldValue, int newValue) {
	int replaced = atomicCAS((int*)ref, oldValue, newValue);
	return replaced == oldValue;
}

__device__ bool tryAtomicStore(volatile float* ref, float oldValue, float newValue) {
	return tryAtomicStore((volatile int*)ref, __float_as_int(oldValue), __float_as_int(newValue));
}