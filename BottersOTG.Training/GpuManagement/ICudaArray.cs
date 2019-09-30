using System;
using ManagedCuda.BasicTypes;

namespace Telogis.RouteCloud.GPUManagement {
	public interface ICudaArray {
		CUdeviceptr DevicePointer { get; }

		SizeT Size { get; }

		SizeT SizeInBytes { get; }

		SizeT TypeSize { get; }

		Type Type { get; }
	}
}
