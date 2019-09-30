#pragma once
#include "GPUConstants.generated.cu"

__device__ float Entropy(const float accuracy) {
	if (accuracy <= 0 || accuracy >= 1) {
		return 0;
	} else {
		return -accuracy * logf(accuracy);
	}
}

__device__ float Entropy(const float* classFrequencies, const float totalWeight) {
	float entropy = 0.0f;
	for (int i = 0; i < Constants_MaxClasses; ++i) {
		entropy += Entropy(totalWeight == 0 ? 0 : classFrequencies[i] / totalWeight);
	}
	return entropy / Constants_MaxClasses;
}