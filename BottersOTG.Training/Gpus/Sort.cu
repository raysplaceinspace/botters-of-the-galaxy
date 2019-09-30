#include "Sort.cuh"
#include "Swap.cuh"


/* maps an index t onto the bitonic network with step size of inc. e.g.
   inc = 0001000
     t = 0111101
          ///|||
result = 1110101 */
__device__
int _getBitonicPosition(int t, int inc) {
	int low = t & (inc - 1); // bits of t below inc [|||]
	return (t << 1) - low; // leftshift upper bits [///] and insert 0 bit at inc
}

template<typename sortkey_t, typename sortvalue_t>
__device__
void bitonicGlobal(sortkey_t *keys, sortvalue_t* values, int size, int inc, int dir) {
	for (int t = threadIdx.x; t < size; t += blockDim.x) {
		int i = _getBitonicPosition(t, inc);
		if (i + inc >= size) {
			break;
		}
		bool groupPatternReverse = (dir & i) == 0; // alternate comparison every dir items
		bool paddingFixReverse = (size & dir) == 0; // make sure we compare DESC when i < size <= i+inc

		sortkey_t k0 = keys[i];
		sortkey_t k1 = keys[i + inc];

		if ((k0 < k1) ^ groupPatternReverse ^ paddingFixReverse) {
			SWAP(sortkey_t, k0, k1);
			SWAP(sortvalue_t, values[i], values[i + inc]);
		}

		keys[i] = k0;
		keys[i + inc] = k1;
	}
}

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
__device__
void sortBitonic(sortkey_t* keys, sortvalue_t* values, int size) {
	for (int block = 1; block < size * 2; block <<= 1) { // 1,2,4,8*,16,32 => dir swaps every 16 values
		for (int inc = block; inc > 0; inc >>= 1) { // 8,4*,2,1
			bitonicGlobal(keys, values, size, inc, block << 1);
			__syncthreads();
		}
	}
}
