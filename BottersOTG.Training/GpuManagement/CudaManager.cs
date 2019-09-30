using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Telogis.RouteCloud.GPUManagement {
	// TODO: consider clearing out modules and/or kernels on CudaManagerPool.Release
	public class CudaManager : IDisposable {
		public CudaContext Context { get; private set; }

		/** <summary>Maps 1-to-1 to the CUDA context bound to the current thread. Should only be used by CudaArray. </summary> */
		[ThreadStatic]
		internal static CudaManager Current = null;

		private readonly Dictionary<string, CudaKernel> _kernels = new Dictionary<string, CudaKernel>();

		private readonly Dictionary<string, CUmodule> _modules = new Dictionary<string, CUmodule>();

		public readonly int OrderId;
		public readonly int DeviceId;
		private readonly Action<CudaManager> _disposeAction;

		private bool _allocationHasFailed = false;

		public AllocationStats AllocationStats { get; private set; }

		///<summary>Do not construct this directly, instead use CudaManagerPool</summary>
		public CudaManager(int orderId, int deviceId, Action<CudaManager> disposeAction) {
			OrderId = orderId;
			DeviceId = deviceId;
			_disposeAction = disposeAction;

			try {
				Context = new CudaContext(deviceId, CUCtxFlags.BlockingSync);
			} catch (Exception e) {
				string message = string.Format("Failed to create a CudaContext with id {0} (deviceId={1}), " +
					"which indicates a system failure rather than request failure", OrderId, DeviceId);
				throw new InvalidOperationException(message, e);
			}
		}

		public void SetCurrent() {
			Context.SetCurrent();
			SetupForThread();
		}

		public void SetInactive() {
			TeardownForThread();
		}

		/// <summary>This won't dispose the CudaContext directly, but the disposeAction supplied in the constructor might</summary>
		public void Dispose() {
			_disposeAction(this);
		}

		public void DisposeCuda() {
			CudaContext context = Context;
			if (context != null) {
				Context = null;
				try {
					context.Dispose();
				} catch {
					// if this fails, it will probably just mask the actual error further up the stack (e.g. OOM)
				}
			}
		}

		public CudaKernel GetOrCreateCudaKernel(string moduleName, string kernelName) {
			CudaKernel kernel;
			moduleName = "MasterInclude";
			if (!_kernels.TryGetValue(kernelName, out kernel)) {

				CUmodule module;
				if (!_modules.TryGetValue(moduleName, out module)) {

					string fatbinName = "";
					if (IntPtr.Size == 8) {
						fatbinName = moduleName + ".x64.fatbin";
					} else {
						fatbinName = moduleName + ".fatbin";
					}

					using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(fatbinName)) {
						if (stream == null) {
							throw new Exception($"Fatbin embedded resource '{fatbinName}' could not be found");
						}
						module = Context.LoadModuleFatBin(stream);
						_modules[moduleName] = module;
					}
				}
				kernel = new CudaKernel(kernelName, module, Context);
				_kernels[kernelName] = kernel;
			}
			return kernel;
		}

		public void MarkAllocationFailure() {
			_allocationHasFailed = true;
		}

		public bool IsValid() {
			if (Context == null || _allocationHasFailed) {
				return false;
			}
			try {
				Context.GetFreeDeviceMemorySize();
				return true;
			} catch {
				return false;
			}
		}

		private void SetupForThread() {
			AllocationStats = new AllocationStats();
			Current = this;
		}

		private void TeardownForThread() {
			Current = null;
			AllocationStats = null;
		}
	}
}
