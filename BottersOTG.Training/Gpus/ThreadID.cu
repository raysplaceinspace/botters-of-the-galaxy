__device__
int get_global_id(int dimension) {
	if(dimension == 0) {
		return blockIdx.x * blockDim.x + threadIdx.x;
	} else if(dimension == 1) {
		return blockIdx.y * blockDim.y + threadIdx.y;
	} else if(dimension == 2) {
		return blockIdx.z * blockDim.z + threadIdx.z;
	} else {
		return 0;
	}
}

__device__
int get_global_size(int dimension) {
	if(dimension == 0) {
		return gridDim.x * blockDim.x;
	} else if (dimension == 1) {
		return gridDim.y * blockDim.y;
	} else if (dimension == 2) {
		return gridDim.z * blockDim.z;
	} else {
		return 0;
	}
}
