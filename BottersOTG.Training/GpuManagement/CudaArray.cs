using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Telogis.RouteCloud.GPUManagement {
	public class CudaArray<T> : IDisposable, ICudaArray where T : struct {
		private CudaDeviceVariable<T> _array;

		/// <summary>the logical size of the array</summary>
		/// <remarks>the actual size of the array may be larger than this (e.g. when the logical size is 0)</remarks>
		public SizeT Size { get; }
		public SizeT TypeSize => _array.TypeSize;
		public SizeT SizeInBytes => Size * TypeSize;
		public Type Type => typeof(T);
		public CUdeviceptr DevicePointer => _array.DevicePointer;

		public CudaArray(SizeT size) {
			try {
				_array = new CudaDeviceVariable<T>(size == 0 ? (SizeT)1 : size);
			} catch (CudaException) {
				CudaManager.Current?.MarkAllocationFailure();
				throw;
			}
			Size = size;
			CudaManager.Current?.AllocationStats?.LogArrayCreate(this);
		}

		public void Dispose() {
			_array.Dispose();
			CudaManager.Current?.AllocationStats?.LogArrayDispose(this);
		}

		// Convert an Array to a CudaArray and write it to the device
		public static implicit operator CudaArray<T>(T[] array) {
			CudaArray<T> cudaArray = new CudaArray<T>(array.LongLength);
			cudaArray.Write(array);
			return cudaArray;
		}

		public T this[SizeT index] => _array[index];

		public T[] Read() {
			if (Size == 0) {
				return new T[0];
			}
			return _array;
		}

		public void Write(T[] array) {
			_array.CopyToDevice(array);
		}

		public void Copy(CudaArray<T> source) {
			_array.AsyncCopyToDevice(source._array, default(CUstream));
		}

		public void CopyFromHost(T[] source) {
			_array.CopyToDevice(source);
		}
	}
}
