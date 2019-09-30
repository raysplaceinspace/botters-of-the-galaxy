#include "../Gpus/Array.cuh"
#include "GPUConstants.generated.cu"
#include "GPUDataPoint.generated.cu"
#include "GPUDecisionLearnerContext.generated.cu"
#include "GPUNode.generated.cu"
#include "GPUSplit.generated.cu"
#include "Entropy.cuh"

__kernel void dlInitContext(
	DecisionLearnerContext* context,
	DataPoint* dataPoints,
	int numDataPoints,
	float totalWeight,
	int numAttributeAxes,
	int numCategoricalAxes,
	int* dataPointIds,
	Node* nodes,
	int* openNodeIds,
	int* nextOpenNodeIds,
	int maxOpenNodeIds) {

	context->DataPoints = dataPoints;
	context->NumDataPoints = numDataPoints;
	context->TotalWeight = totalWeight;
	context->NumAttributeAxes = numAttributeAxes;
	context->NumCategoricalAxes = numCategoricalAxes;
	context->DataPointIds = dataPointIds;
	context->Nodes = nodes;
	context->CurrentLevel = -1;
	context->OpenNodeIds = openNodeIds;
	context->NumOpenNodes = 0;
	context->NextOpenNodeIds = nextOpenNodeIds;
	context->NumNextOpenNodes = 0;
	context->MaxOpenNodes = maxOpenNodeIds;
}

__kernel void dlInitDataPoints(DecisionLearnerContext* context) {
	int i = get_global_id(0);
	if (i < context->NumDataPoints) {
		context->DataPointIds[i] = i;
	}
}

__kernel void dlInitialNode(DecisionLearnerContext* context) {
	if (get_global_id(0) > 0) {
		return;
	}

	// Create initial node
	Node* __restrict__ node = &context->Nodes[0];
	node->RangeStart = 0;
	node->RangeLength = context->NumDataPoints;
	context->CurrentLevel = 0;

	// Open this node
	context->OpenNodeIds[0] = 0;
	context->NumOpenNodes = 1;
}

__kernel void dlLeafOpenNodes(DecisionLearnerContext* context) {
	// assert(blockDim.x == Constants_NumThreadsPerBlock);
	// assert(blockDim.x >= Constants_MaxClasses);

	__shared__ float allClassFrequenciesBuffer[Constants_NumThreadsPerBlock * Constants_MaxClasses];
	Array2D<float> allClassFrequencies(allClassFrequenciesBuffer, Constants_NumThreadsPerBlock, Constants_MaxClasses);

	const Array<int> allDataPointIds(context->DataPointIds, context->NumDataPoints);

	for (int i = blockIdx.x; i < context->NumOpenNodes; i += gridDim.x) {
		const int nodeId = context->OpenNodeIds[i];
		if (nodeId == -1) {
			continue;
		}
		Node* __restrict__ node = &context->Nodes[nodeId];

		// Initialise stripeFrequencies

		// Calculate the stripe sum of class frequencies
		{
			float* __restrict__ stripeFrequencies = &allClassFrequencies.at(threadIdx.x, 0);
			for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
				stripeFrequencies[classId] = 0.0f;
			}
			// Don't need to sync threads because only changed the part of the array owned by this thread

			for (int col = threadIdx.x; col < node->RangeLength; col += blockDim.x) {
				const int dataPointId = allDataPointIds.at(node->RangeStart + col);
				const DataPoint* dataPoint = &context->DataPoints[dataPointId];
				stripeFrequencies[dataPoint->Class] += dataPoint->Weight;
			}
		}
		__syncthreads();
		
		// Combine stripes
		const int classId = threadIdx.x;
		if (classId < Constants_MaxClasses) {
			float* __restrict__ totalFrequencies = &allClassFrequencies.at(0, 0);
			for (int k = 1; k < allClassFrequencies.length0(); ++k) {
				const float* stripeFrequencies = &allClassFrequencies.at(k, 0);
				totalFrequencies[classId] += stripeFrequencies[classId];
			}
		}
		__syncthreads();

		// Write result
		if (threadIdx.x == 0) {
			const float* totalFrequencies = &allClassFrequencies.at(0, 0);
			float totalWeight = 0.0f;
			float bestClassWeight = 0.0f;
			for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
				const float classWeight = totalFrequencies[classId];
				if (classWeight > bestClassWeight) {
					bestClassWeight = classWeight;
				}
				totalWeight += classWeight;
			}

			const float entropy = Entropy(totalFrequencies, totalWeight);

			for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
				node->ClassDistribution[classId] = totalFrequencies[classId];
			}
			node->Entropy = entropy;
			node->TotalWeight = totalWeight;

			node->SplitType = Constants_SplitType_None;
			node->LeftChild = -1;
			node->RightChild = -1;

			const float maximumImprovement = totalWeight - bestClassWeight;
			const float requiredImprovement = context->TotalWeight * Constants_RequiredImprovementToSplit;
			if (maximumImprovement < requiredImprovement) {
				// Remove this node from the open node list
				context->OpenNodeIds[i] = -1;
			}
		}
	}
}

__kernel void dlFindOptimalSplitPerNode(DecisionLearnerContext* context, const Split* allSplitsBuffer, int numSplits, Split* bestSplitsBuffer) {
	const Array2D<Split> allSplits(allSplitsBuffer, context->MaxOpenNodes, Constants_MaxSplits);
	Array<Split> bestSplits(bestSplitsBuffer, context->MaxOpenNodes);

	for (int openNodeIndex = get_global_id(0); openNodeIndex < context->NumOpenNodes; openNodeIndex += get_global_size(0)) {
		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		const Node* node = &context->Nodes[nodeId];

		const Split* bestSplit = nullptr;
		for (int splitId = 0; splitId < numSplits; ++splitId) {
			const Split* split = &allSplits.at(openNodeIndex, splitId);
			if (bestSplit == nullptr || split->Entropy < bestSplit->Entropy) {
				bestSplit = split;
			}
		}

		Split* output = &bestSplits.at(openNodeIndex); 
		if (bestSplit != nullptr && bestSplit->Entropy < node->Entropy) {
			*output = *bestSplit;
		} else {
			output->SplitType = Constants_SplitType_Null;
		}
	}
}

__kernel void dlNextLevel(DecisionLearnerContext* context) {
	for (int openNodeIndex = get_global_id(0); openNodeIndex < context->NumNextOpenNodes; openNodeIndex += get_global_size(0)) {
		context->OpenNodeIds[openNodeIndex] = context->NextOpenNodeIds[openNodeIndex];
	}

	if (get_global_id(0) == 0) {
		context->NumOpenNodes = context->NumNextOpenNodes;
		context->NumNextOpenNodes = 0;
		++context->CurrentLevel;
	}
}