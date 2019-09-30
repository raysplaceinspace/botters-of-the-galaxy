#include <assert.h>
#include "../../Gpus/Array.cuh"
#include "../../Gpus/Sort.cuh"
#include "../GPUConstants.generated.cu"
#include "../GPUDataPoint.generated.cu"
#include "../GPUSplit.generated.cu"
#include "GPUAttributeDataPoint.generated.cu"
#include "../Entropy.cuh"

__device__ float SplitEntropy(
	const float* leftFrequencies,
	const float* totalFrequencies,
	const float totalLeft,
	const float totalRight) {
	
	float entropyLeft = 0.0f;
	float entropyRight = 0.0f;
	for (int i = 0; i < Constants_MaxClasses; ++i) {
		const float leftFrequency = leftFrequencies[i];
		const float rightFrequency = totalFrequencies[i] - leftFrequencies[i];
		entropyLeft += Entropy(totalLeft == 0 ? 0 : leftFrequency / totalLeft);
		entropyRight += Entropy(totalRight == 0 ? 0 : rightFrequency / totalRight);
	}
	entropyLeft /= Constants_MaxClasses;
	entropyRight /= Constants_MaxClasses;

	const float total = totalLeft + totalRight;
	const float entropy =
		entropyLeft * (totalLeft / total) +
		entropyRight * (totalRight / total);
	return entropy;
}

__kernel void spaCopyDataPointsPerAxis(const DecisionLearnerContext* context, AttributeDataPoint* attributePointsBuffer, float* sortKeysBuffer) {
	Array2D<AttributeDataPoint> allAttributePoints(attributePointsBuffer, context->NumAttributeAxes, context->NumDataPoints);
	Array2D<float> allSortKeys(sortKeysBuffer, context->NumAttributeAxes, context->NumDataPoints);
	Array<int> dataPointIds(context->DataPointIds, context->NumDataPoints);

	for (int openNodeIndex = blockIdx.x; openNodeIndex < context->NumOpenNodes; openNodeIndex += gridDim.x) {
		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		const Node* node = &context->Nodes[nodeId];

		for (int col = threadIdx.x; col < node->RangeLength; col += blockDim.x) {
			const int dataPointId = dataPointIds.at(node->RangeStart + col);
			const DataPoint* dataPoint = &context->DataPoints[dataPointId];

			for (int axisId = 0; axisId < context->NumAttributeAxes; ++axisId) {
				AttributeDataPoint* __restrict__ attributePoint = &allAttributePoints.at(axisId, node->RangeStart + col);
				float* __restrict__ sortKey = &allSortKeys.at(axisId, node->RangeStart + col);

				attributePoint->DataPointId = dataPointId;
				attributePoint->Weight = dataPoint->Weight;
				attributePoint->Class = dataPoint->Class;
				attributePoint->Attribute = dataPoint->AllAttributes[axisId];
				*sortKey = -attributePoint->Attribute; // negative to make ascending, because Sort.cuh is descending
			}
		}
	}
}

__kernel void spaSortDataPointsPerAxis(const DecisionLearnerContext* context, AttributeDataPoint* attributePointsBuffer, float* sortKeysBuffer) {
	Array2D<AttributeDataPoint> allAttributePoints(attributePointsBuffer, context->NumAttributeAxes, context->NumDataPoints);
	Array2D<float> allSortKeys(sortKeysBuffer, context->NumAttributeAxes, context->NumDataPoints);

	const int numBlocks = context->NumAttributeAxes * context->NumOpenNodes;
	for (int i = blockIdx.x; i < numBlocks; i += gridDim.x) {
		const int openNodeIndex = i / context->NumAttributeAxes;
		const int axisId = i % context->NumAttributeAxes;

		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		const Node* node = &context->Nodes[nodeId];

		AttributeDataPoint* axisPoints = &allAttributePoints.at(axisId, node->RangeStart);
		float* sortKeys = &allSortKeys.at(axisId, node->RangeStart);

		sortBitonic(sortKeys, axisPoints, node->RangeLength);
	}
}

__kernel void spaAccumulateFrequenciesPerAxis(const DecisionLearnerContext* context, AttributeDataPoint* attributePointsBuffer, float* cumulativeFrequenciesBuffer) {
	Array2D<AttributeDataPoint> allAttributePoints(attributePointsBuffer, context->NumAttributeAxes, context->NumDataPoints);
	Array3D<float> allCumulativeFrequencies(cumulativeFrequenciesBuffer, context->NumAttributeAxes, context->NumDataPoints, Constants_MaxClasses);

	const int numBlocks = context->NumAttributeAxes * context->NumOpenNodes;
	for (int i = blockIdx.x; i < numBlocks; i += gridDim.x) {
		const int openNodeIndex = i / context->NumAttributeAxes;
		const int axisId = i % context->NumAttributeAxes;

		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		const Node* node = &context->Nodes[nodeId];
		
		// Initialise cumulative frequencies
		for (int col = threadIdx.x; col < node->RangeLength; col += blockDim.x) {
			for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
				const AttributeDataPoint* attributePoint = &allAttributePoints.at(axisId, node->RangeStart + col);
				float* __restrict__ cumulativeFrequency = &allCumulativeFrequencies.at(axisId, node->RangeStart + col, classId);
				*cumulativeFrequency = attributePoint->Class == classId ? attributePoint->Weight : 0.0f;
			}
		}
		__syncthreads();

		// Accumulative frequencies
		for (int halfSectionWidth = 1; halfSectionWidth < node->RangeLength; halfSectionWidth *= 2) {
			for (int addToCol = threadIdx.x; addToCol < node->RangeLength; addToCol += blockDim.x) {
				int addFromCol = -1;
				{
					/*
					Step 1: add 0 to 1, 2 to 3, 4 to 5, 6 to 7
					Step 2: add 1 to 2, 1 to 3, 5 to 6, 5 to 7
					Step 3: add 3 to 4, 3 to 5, 3 to 6, 3 to 7
					*/
					const int sectionWidth = halfSectionWidth * 2;
					const int sectionStart = (addToCol / sectionWidth) * sectionWidth; // integer division
					const int sectionOffset = halfSectionWidth - 1; // top of the bottom half
					const int newFromCol = sectionStart + sectionOffset;
					if (newFromCol < addToCol) {
						addFromCol = newFromCol;
					}
				}
				if (addFromCol != -1) {
					for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
						float other = allCumulativeFrequencies.at(axisId, node->RangeStart + addFromCol, classId);
						float* __restrict__ cumulativeFrequency = &allCumulativeFrequencies.at(axisId, node->RangeStart + addToCol, classId);
						*cumulativeFrequency += other;
					}
				}
			}
			__syncthreads();
		}
	}
}

__kernel void spaBestSplitPerAxis(
	const DecisionLearnerContext* context,
	const AttributeDataPoint* attributePointsBuffer,
	const float* cumulativeFrequenciesBuffer,
	Split* splits,
	const int splitWriteStart) {

	const float Precision = 0.001f;

	__shared__ int bestColPerThread[Constants_NumThreadsPerBlock];
	__shared__ float bestEntropyPerThread[Constants_NumThreadsPerBlock];

	const Array2D<AttributeDataPoint> allAttributePoints(attributePointsBuffer, context->NumAttributeAxes, context->NumDataPoints);
	const Array3D<float> allCumulativeFrequencies(cumulativeFrequenciesBuffer, context->NumAttributeAxes, context->NumDataPoints, Constants_MaxClasses);
	Array2D<Split> allSplits(splits, context->MaxOpenNodes, Constants_MaxSplits);

	const int numBlocks = context->NumAttributeAxes * context->NumOpenNodes;
	for (int i = blockIdx.x; i < numBlocks; i += gridDim.x) {
		const int openNodeIndex = i / context->NumAttributeAxes;
		const int axisId = i % context->NumAttributeAxes;

		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		const Node* node = &context->Nodes[nodeId];

		// Calculate stripe best split entropy
		{
			int bestCol = -1;
			float bestEntropy = node->Entropy;

			const int lastCol = node->RangeLength - 1;
			const float* totalFrequencies = &allCumulativeFrequencies.at(axisId, node->RangeStart + lastCol, 0);
			for (int col = threadIdx.x; col < node->RangeLength; col += blockDim.x) {
				if (col == 0) {
					// this is a pointless split (left is empty)
					continue;
				} else if (allAttributePoints.at(axisId, node->RangeStart + col - 1).Attribute == allAttributePoints.at(axisId, node->RangeStart + col).Attribute) {
					// Can't make an inequality split the same attribute value
					continue;
				}

				// right starts from col, left starts from (col - 1)
				const float* leftFrequencies = &allCumulativeFrequencies.at(axisId, node->RangeStart + col - 1, 0); 
				float totalLeft = 0.0f;
				float total = 0.0f;
				for (int i = 0; i < Constants_MaxClasses; ++i) {
					totalLeft += leftFrequencies[i];
					total += totalFrequencies[i];
				}
				float totalRight = total - totalLeft;

				if (totalLeft <= Precision || totalRight <= Precision) {
					// This is a non-split (left or right empty)
				} else {
					float entropy = SplitEntropy(leftFrequencies, totalFrequencies, totalLeft, totalRight);
					assert(entropy <= node->Entropy + Precision);
					if (entropy < bestEntropy) {
						bestCol = col;
						bestEntropy = entropy;
					}
				}
			}

			bestColPerThread[threadIdx.x] = bestCol;
			bestEntropyPerThread[threadIdx.x] = bestEntropy;
		}
		__syncthreads();

		// Calculate overall best split entropy
		if (threadIdx.x == 0) {
			int bestCol = -1;
			float bestEntropy = node->Entropy;

			for (int i = 0; i < blockDim.x; ++i) {
				if (bestEntropyPerThread[i] < bestEntropy) {
					bestCol = bestColPerThread[i];
					bestEntropy = bestEntropyPerThread[i];
				}
			}

			Split* __restrict__ split = &allSplits.at(openNodeIndex, splitWriteStart + axisId);
			if (bestCol != -1) {
				const AttributeDataPoint* bestSplitPoint = &allAttributePoints.at(axisId, node->RangeStart + bestCol);

				split->Entropy = bestEntropy;
				split->SplitType = Constants_SplitType_Attribute;
				split->Axis = axisId;
				split->Column = bestCol;
				split->SplitCategories = 999;
				split->SplitAttribute = bestSplitPoint->Attribute;
			} else {
				split->Entropy = 999;
				split->SplitType = Constants_SplitType_Null;
				split->Axis = 0xFF;
				split->Column = -1;
				split->SplitCategories = 999;
				split->SplitAttribute = 999;
			}
		}
	}
}

__kernel void spaApplyOptimalSplit(
	DecisionLearnerContext* context,
	const AttributeDataPoint* attributePointsBuffer,
	const Split* bestSplitsBuffer) {

	const Array2D<AttributeDataPoint> allAttributePoints(attributePointsBuffer, context->NumAttributeAxes, context->NumDataPoints);
	const Array<Split> bestSplits(bestSplitsBuffer, context->MaxOpenNodes);
	Array<int> allDataPointIds(context->DataPointIds, context->NumDataPoints);

	for (int openNodeIndex = blockIdx.x; openNodeIndex < context->NumOpenNodes; openNodeIndex += gridDim.x) {
		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		Node* __restrict__ node = &context->Nodes[nodeId];

		const Split* bestSplit = &bestSplits.at(openNodeIndex);
		if (bestSplit->SplitType != Constants_SplitType_Attribute) {
			continue;
		}
		const int axisId = bestSplit->Axis;

		// We've already sorted this axis, just copy the sorted data point IDs to the right place
		for (int i = threadIdx.x; i < node->RangeLength; i += blockDim.x) {
			const AttributeDataPoint* attributePoint = &allAttributePoints.at(axisId, node->RangeStart + i);
			int* __restrict__ dataPointId = &allDataPointIds.at(node->RangeStart + i);
			*dataPointId = attributePoint->DataPointId;
		}

		if (threadIdx.x == 0) {
			// Update parent split fields
			node->SplitType = Constants_SplitType_Attribute;
			node->SplitAttribute = bestSplit->SplitAttribute;
			node->SplitCategories = 999;
			node->SplitAxis = bestSplit->Axis;
			node->LeftChild = nodeId * 2 + 1;
			node->RightChild = nodeId * 2 + 2;

			// Write new child nodes
			Node* __restrict__ leftNode = &context->Nodes[node->LeftChild];
			Node* __restrict__ rightNode = &context->Nodes[node->RightChild];

			leftNode->RangeStart = node->RangeStart;
			leftNode->RangeLength = bestSplit->Column;
			rightNode->RangeStart = node->RangeStart + bestSplit->Column;
			rightNode->RangeLength = node->RangeLength - bestSplit->Column;

			// Add to next open list
			const int leftOpenIndex = atomicAdd(&context->NumNextOpenNodes, 2);
			const int rightOpenIndex = leftOpenIndex + 1;
			int* __restrict__ nextOpenNodes = context->NextOpenNodeIds;
			nextOpenNodes[leftOpenIndex] = node->LeftChild;
			nextOpenNodes[rightOpenIndex] = node->RightChild;
		}
	}
}