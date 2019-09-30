#pragma once

/*
Does an in-place sort (DESC) on the keys, and a corresponding sort on the values.
The keys are floats, and the values are shorts (e.g. indexes into an array).
size is the length of each of the vectors.

Each sort needs to be in a single work-group, and the work-group size should be
1024. Hence if multiple sorts are needed at once, you can queue up a single kernel
with numListsToSort * numWorkGroups threads, and use the correct vector based on
get_group_id(0).
*/
template<typename sortkey_t, typename sortvalue_t>
__device__ void sortBitonic(sortkey_t* keys, sortvalue_t* values, int size);