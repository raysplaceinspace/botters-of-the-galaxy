#include <math_constants.h>
#include <stdio.h>
#include "../../Gpus/Array.cuh"
#include "../GPUConstants.generated.cu"
#include "../GPUDataPoint.generated.cu"
#include "../GPUSplit.generated.cu"
#include "GPUCategoricalDataPoint.generated.cu"
#include "../Entropy.cuh"

__device__
uint CategoryBitMask(const int categoryId) {
	return ((uint)1) << categoryId;
}

__device__
float IncrementalEntropy(
	Array<float> classDistributionRight,
	Array<float> additionalRight,
	Array<float> totalClassDistribution,
	const float totalLeft,
	const float totalRight) {

	float entropyLeft = 0.0f;
	float entropyRight = 0.0f;
	for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
		const float rightFrequency = classDistributionRight.at(classId) + additionalRight.at(classId);
		const float leftFrequency = totalClassDistribution.at(classId) - rightFrequency;
		
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

__kernel void spcCopyDataPointsPerAxis(const DecisionLearnerContext* context, CategoricalDataPoint* categoricalPointsBuffer) {
	Array2D<CategoricalDataPoint> allCategoricalPoints(categoricalPointsBuffer, context->NumCategoricalAxes, context->NumDataPoints);
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

			for (int axisId = 0; axisId < context->NumCategoricalAxes; ++axisId) {
				CategoricalDataPoint* __restrict__ categoricalPoint = &allCategoricalPoints.at(axisId, node->RangeStart + col);

				categoricalPoint->DataPointId = dataPointId;
				categoricalPoint->Weight = dataPoint->Weight;
				categoricalPoint->Class = dataPoint->Class;
				categoricalPoint->Categories = dataPoint->AllCategories[axisId];
			}
		}
	}
}

__device__ void spcCalculateBestSplitForNodeAxis(
	const DecisionLearnerContext* context,
	const Node* node,
	const int axisId,
	const Array2D<CategoricalDataPoint> allCategoricalPoints,
	Split* __restrict__ split) {

	const float Precision = 0.001f;

	__shared__ float incrementalFrequenciesBuffer[Constants_MaxCategories * Constants_MaxClasses];
	__shared__ float incrementalEntropyPerCategoryBuffer[Constants_MaxCategories];
	__shared__ float bestIncrementalEntropy;
	__shared__ float bestIncrementalCategory;

	__shared__ float classDistributionRightBuffer[Constants_MaxClasses];
	__shared__ uint categoriesRight;

	__shared__ float bestOverallEntropy;
	__shared__ uint bestOverallCategories;

	Array2D<float> incrementalFrequenciesPerCategory(incrementalFrequenciesBuffer, Constants_MaxCategories, Constants_MaxClasses);
	Array<float> incrementalEntropyPerCategory(incrementalEntropyPerCategoryBuffer, Constants_MaxCategories);

	Array<float> classDistributionRight(classDistributionRightBuffer, Constants_MaxClasses);

	const Array<float> totalClassDistribution(node->ClassDistribution, Constants_MaxClasses);
	
	// Zero frequencies
	if (threadIdx.x == 0) {
		categoriesRight = 0;
		bestOverallEntropy = node->Entropy;
		bestOverallCategories = 0;
	}
	for (int classId = threadIdx.x; classId < Constants_MaxClasses; classId += blockDim.x) {
		classDistributionRight.at(classId) = 0.0f;
	}
	__syncthreads();

	for (int addCategoryIteration = 0; addCategoryIteration < Constants_MaxCategories; ++addCategoryIteration) {
		// Initialize additional frequencies with zeroes
		for (int categoryId = threadIdx.x; categoryId < Constants_MaxCategories; categoryId += blockDim.x) {
			incrementalEntropyPerCategory.at(categoryId) = CUDART_INF_F;
			for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
				incrementalFrequenciesPerCategory.at(categoryId, classId) = 0.0f;
			}
		}
		__syncthreads();

		// For each category, what would the effect be on the weight distribution if this category was moved to the right?
		for (int col = threadIdx.x; col < node->RangeLength; col += blockDim.x) {
			const CategoricalDataPoint* categoricalPoint = &allCategoricalPoints.at(axisId, node->RangeStart + col);
			if ((categoricalPoint->Categories & categoriesRight) != 0) {
				// This data point is already on the right, can't add it again
				continue;
			}

			for (int categoryId = 0; categoryId < Constants_MaxCategories; ++categoryId) {
				uint categoryBitMask = CategoryBitMask(categoryId);
				if ((categoryBitMask & categoriesRight) != 0) {
					// This category is already on the right, can't add it again
					continue;
				}

				if ((categoricalPoint->Categories & categoryBitMask) != 0) {
					float* __restrict__ frequency = &incrementalFrequenciesPerCategory.at(categoryId, categoricalPoint->Class);
					atomicAdd(frequency, categoricalPoint->Weight);
				}
			}
		}
		__syncthreads();

		// For each category, if it was added, what would the new entropy be?
		for (int categoryId = threadIdx.x; categoryId < Constants_MaxCategories; categoryId += blockDim.x) {
			uint categoryBitMask = CategoryBitMask(categoryId);
			if ((categoryBitMask & categoriesRight) != 0) { continue; }

			Array<float> additionalRight = incrementalFrequenciesPerCategory.slice(categoryId);

			float total = 0.0f;
			float totalRight = 0.0f;
			float totalAddition = 0.0f;
			for (int classId = 0; classId < Constants_MaxClasses; ++classId) {
				total += totalClassDistribution.at(classId);
				totalRight += classDistributionRight.at(classId) + additionalRight.at(classId);

				totalAddition += additionalRight.at(classId);
			}
			const float totalLeft = total - totalRight;

			if (totalAddition <= Precision || totalLeft <= Precision || totalRight <= Precision) {
				// This split achieves nothing
				continue;
			} else {
				float* __restrict__ incrementalEntropy = &incrementalEntropyPerCategory.at(categoryId);
				*incrementalEntropy = IncrementalEntropy(classDistributionRight, additionalRight, totalClassDistribution, totalLeft, totalRight);
				assert(*incrementalEntropy <= node->Entropy + Precision);
			}
		}
		__syncthreads();

		// Which additional category has the best entropy?
		if (threadIdx.x == 0) {
			bestIncrementalEntropy = CUDART_INF_F;
			bestIncrementalCategory = -1;
			for (int categoryId = 0; categoryId < Constants_MaxCategories; ++categoryId) {
				uint categoryBitMask = CategoryBitMask(categoryId);
				if ((categoryBitMask & categoriesRight) != 0) { continue; }

				const float entropy = incrementalEntropyPerCategory.at(categoryId);
				if (entropy < bestIncrementalEntropy) {
					bestIncrementalEntropy = entropy;
					bestIncrementalCategory = categoryId;
				}
			}
		}
		__syncthreads();

		// Add new incremental category
		if (bestIncrementalCategory != -1) {
			const Array<float> additionalRight = incrementalFrequenciesPerCategory.slice(bestIncrementalCategory);
			for (int classId = threadIdx.x; classId < Constants_MaxClasses; classId += blockDim.x) {
				float* __restrict__ rightFrequency = &classDistributionRight.at(classId);
				*rightFrequency += additionalRight.at(classId);
			}
			if (threadIdx.x == 0) {
				categoriesRight |= CategoryBitMask(bestIncrementalCategory);

				if (bestIncrementalEntropy < bestOverallEntropy) {
					bestOverallEntropy = bestIncrementalEntropy;
					bestOverallCategories = categoriesRight;
				}
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		assert(bestOverallEntropy <= node->Entropy);
		if (bestOverallCategories != 0) {
			split->Entropy = bestOverallEntropy;
			split->SplitType = Constants_SplitType_Categorical;
			split->Axis = axisId;
			split->Column = -1;
			split->SplitCategories = bestOverallCategories;
			split->SplitAttribute = 999;
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

__kernel void spcBestCategoricalSplitPerAxis(
	const DecisionLearnerContext* context,
	CategoricalDataPoint* categoricalPointsBuffer,
	Split* splits,
	int splitWriteStart) {

	Array2D<CategoricalDataPoint> allCategoricalPoints(categoricalPointsBuffer, context->NumCategoricalAxes, context->NumDataPoints);
	Array2D<Split> allSplits(splits, context->MaxOpenNodes, Constants_MaxSplits);

	const int numBlocks = context->NumCategoricalAxes * context->NumOpenNodes;
	for (int i = blockIdx.x; i < numBlocks; i += gridDim.x) {
		const int openNodeIndex = i / context->NumCategoricalAxes;
		const int axisId = i % context->NumCategoricalAxes;

		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		const Node* node = &context->Nodes[nodeId];
		Split* __restrict__ split = &allSplits.at(openNodeIndex, splitWriteStart + axisId);
		spcCalculateBestSplitForNodeAxis(context, node, axisId, allCategoricalPoints, split);
	}
}

__device__
void spcApplyOptimalSplitToNode(
	DecisionLearnerContext* context,
	Array2D<CategoricalDataPoint> allCategoricalPoints,
	const Split* bestSplit,
	const int nodeId,
	Node* __restrict__ node,
	uint8_t* allSortKeys) {

	__shared__ int splitColumn;
	Array<int> allDataPointIds(context->DataPointIds, context->NumDataPoints);

	const int axisId = bestSplit->Axis;

	// Split data point IDs into left and right
	for (int col = threadIdx.x; col < node->RangeLength; col += blockDim.x) {
		const CategoricalDataPoint* categoricalPoint = &allCategoricalPoints.at(axisId, node->RangeStart + col);
		allSortKeys[node->RangeStart + col] = (categoricalPoint->Categories & bestSplit->SplitCategories) ? 0 : 1; // This order because sort is descending and the sort value is a unsigned byte
	}
	__syncthreads();

	sortBitonic(&allSortKeys[node->RangeStart], &allDataPointIds.at(node->RangeStart), node->RangeLength);

	// Find where the split point is
	if (threadIdx.x == 0) {
		splitColumn = -1;
	}
	__syncthreads();

	for (int col = threadIdx.x; col < node->RangeLength; col += blockDim.x) {
		if (col == 0) {
			continue;
		} else if (allSortKeys[node->RangeStart + col - 1] != allSortKeys[node->RangeStart + col]) {
			splitColumn = col;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && splitColumn != -1) {
		// Update parent split fields
		node->SplitType = Constants_SplitType_Categorical;
		node->SplitAttribute = 999;
		node->SplitCategories = bestSplit->SplitCategories;
		node->SplitAxis = bestSplit->Axis;
		node->LeftChild = nodeId * 2 + 1;
		node->RightChild = nodeId * 2 + 2;

		// Write new child nodes
		Node* __restrict__ leftNode = &context->Nodes[node->LeftChild];
		Node* __restrict__ rightNode = &context->Nodes[node->RightChild];

		leftNode->RangeStart = node->RangeStart;
		leftNode->RangeLength = splitColumn;
		rightNode->RangeStart = node->RangeStart + splitColumn;
		rightNode->RangeLength = node->RangeLength - splitColumn;

		// Add to next open list
		const int leftOpenIndex = atomicAdd(&context->NumNextOpenNodes, 2);
		const int rightOpenIndex = leftOpenIndex + 1;
		int* __restrict__ nextOpenNodes = context->NextOpenNodeIds;
		nextOpenNodes[leftOpenIndex] = node->LeftChild;
		nextOpenNodes[rightOpenIndex] = node->RightChild;
	}
}

__kernel void spcApplyOptimalSplit(
	DecisionLearnerContext* context,
	const CategoricalDataPoint* categoricalPointsBuffer,
	const Split* bestSplitsBuffer,
	uint8_t* sortKeys) {

	Array2D<CategoricalDataPoint> allCategoricalPoints(categoricalPointsBuffer, context->NumCategoricalAxes, context->NumDataPoints);

	const Array<Split> bestSplits(bestSplitsBuffer, context->MaxOpenNodes);

	for (int openNodeIndex = blockIdx.x; openNodeIndex < context->NumOpenNodes; openNodeIndex += gridDim.x) {
		const int nodeId = context->OpenNodeIds[openNodeIndex];
		if (nodeId == -1) {
			continue;
		}
		Node* __restrict__ node = &context->Nodes[nodeId];

		const Split* bestSplit = &bestSplits.at(openNodeIndex);
		if (bestSplit->SplitType != Constants_SplitType_Categorical) {
			continue;
		}
		spcApplyOptimalSplitToNode(context, allCategoricalPoints, bestSplit, nodeId, node, sortKeys);
	}
}