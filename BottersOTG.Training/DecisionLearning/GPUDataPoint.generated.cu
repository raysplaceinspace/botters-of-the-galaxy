// This is a generated file, do not edit it!
#pragma once
#include <stdint.h>
typedef struct DataPoint {
	float Weight;
	uint8_t Class;
	float AllAttributes[Constants_MaxAttributeAxes];
	uint32_t AllCategories[Constants_MaxCategoricalAxes];
} DataPoint;
