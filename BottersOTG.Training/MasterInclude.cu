#define __kernel extern "C" __global__

#include "Gpus/ThreadID.cu"
#include "Gpus/Atomic.cu"
#include "Gpus/Sort.cu"
#include "Gpus/Reduce.cu"

#include "Gpus/GPURandom/GPURandom.cu"

#include "DecisionLearning/DecisionLearner.cu"
#include "DecisionLearning/AttributeSplitting/AttributeSplitter.cu"
#include "DecisionLearning/CategoricalSplitting/CategoricalSplitter.cu"
