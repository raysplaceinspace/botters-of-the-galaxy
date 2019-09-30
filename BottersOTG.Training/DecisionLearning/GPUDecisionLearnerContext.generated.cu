// This is a generated file, do not edit it!
#pragma once
#include <stdint.h>
typedef struct DataPoint DataPoint;
typedef struct Node Node;
typedef struct DecisionLearnerContext {
	DataPoint *DataPoints;
	int32_t NumDataPoints;
	float TotalWeight;
	int32_t NumAttributeAxes;
	int32_t NumCategoricalAxes;
	int32_t *DataPointIds;
	Node *Nodes;
	int32_t CurrentLevel;
	int32_t *OpenNodeIds;
	int32_t NumOpenNodes;
	int32_t *NextOpenNodeIds;
	int32_t NumNextOpenNodes;
	int32_t MaxOpenNodes;
} DecisionLearnerContext;
