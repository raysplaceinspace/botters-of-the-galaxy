using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Telogis.RouteCloud.GPUManagement {
	public class KernelRunner {
		public KernelRunner(KernelManager kernelManager, CudaKernel cudaKernel) {
			_manager = kernelManager;
			Kernel = cudaKernel;
			_extraArguments = new object[0];
		}

		// Only public get because it's a pain to copy SetConstantVariable to this class
		public CudaKernel Kernel { get; private set; }

		private KernelManager _manager;
		private object[] _extraArguments;

		private IntPtr[] _permanentParamsList;
		private GCHandle[] _permanentGCHandleList;
		private CUResult _result;

		public KernelRunner Arguments(params object[] extraArguments) {
			_extraArguments = extraArguments;
			return this;
		}

		private void GetArgsAndSharedMem(out object[] kernelArgs, out uint sharedMem) {
			List<object> kernelParams = new List<object>();
			List<SharedBuffer> sharedParams = new List<SharedBuffer>();

			var args = _manager.PrefixArguments.Concat(_extraArguments).ToArray();
			for (int i = 0; i < args.Length; i++) {
				if (args[i] is ICudaArray) {
					kernelParams.Add(((ICudaArray)args[i]).DevicePointer);
				} else if (args[i] == null) {
					kernelParams.Add(default(CUdeviceptr));
				} else if (args[i].GetType() == typeof(SharedBuffer)) {
					sharedParams.Add((SharedBuffer)args[i]);
				} else {
					kernelParams.Add(args[i]);
				}
			}

			kernelArgs = kernelParams.ToArray();
			sharedMem = (uint)sharedParams.Sum(s => s.Size);
		}

		private void SetKernelDimensions(int numThreads, int numThreadsPerBlock = 0) {
			numThreads = Math.Max(1, numThreads);

			if (numThreadsPerBlock == 0) {
				numThreadsPerBlock = _manager.BlockSize > 0 ? _manager.BlockSize : 32;
			}

			Kernel.BlockDimensions = numThreadsPerBlock;
			Kernel.GridDimensions = (numThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
		}

		public void SetPermanentRunParameters() {
			object[] kernelArgs;
			uint sharedMem;
			GetArgsAndSharedMem(out kernelArgs, out sharedMem);
			Kernel.DynamicSharedMemory = sharedMem;

			int paramCount = kernelArgs.Length;
			_permanentParamsList = new IntPtr[paramCount];
			_permanentGCHandleList = new GCHandle[paramCount];

			// Get pointers to kernel parameters
			for (int i = 0; i < paramCount; i++) {
				_permanentGCHandleList[i] = GCHandle.Alloc(kernelArgs[i], GCHandleType.Pinned);
				_permanentParamsList[i] = _permanentGCHandleList[i].AddrOfPinnedObject();
			}
		}

		public void ExecuteUsingExisting(int numThreads, int numThreadsPerBlock = 0) {
			SetKernelDimensions(numThreads, numThreadsPerBlock);

			_result = DriverAPINativeMethods.Launch.cuLaunchKernel(
				Kernel.CUFunction,
				Kernel.GridDimensions.x,
				Kernel.GridDimensions.y,
				Kernel.GridDimensions.z,
				Kernel.BlockDimensions.x,
				Kernel.BlockDimensions.y,
				Kernel.BlockDimensions.z,
				Kernel.DynamicSharedMemory,
				default(CUstream),
				_permanentParamsList,
				null);
			if (_result != CUResult.Success) {
				throw new CudaException(_result);
			}
		}

		public void Execute(int numThreads, int numThreadsPerBlock = 0) {
			object[] kernelArgs;
			uint sharedMem;

			SetKernelDimensions(numThreads, numThreadsPerBlock);

			GetArgsAndSharedMem(out kernelArgs, out sharedMem);

			Kernel.DynamicSharedMemory = sharedMem;
			Kernel.RunAsync(default(CUstream), kernelArgs);

			if (KernelManager.SynchroniseAfterEveryKernel) {
				_manager.CudaManager.Context.Synchronize();
			}
		}

		public void ExecuteTask() {
			Execute(1, 1);
		}

		public void FreePermanentArguments() {
			// Free pinned managed parameters
			for (int i = 0; i < _permanentGCHandleList.Length; i++) {
				_permanentGCHandleList[i].Free();
			}
		}
	}
}