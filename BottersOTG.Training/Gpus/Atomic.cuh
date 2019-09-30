#pragma once

__device__ bool tryAtomicStore(volatile int* ref, int oldValue, int newValue);

__device__ bool tryAtomicStore(volatile float* ref, float oldValue, float newValue);