using System.Collections.Generic;

namespace Telogis.RouteCloud.GPUManagement {
	public class KernelManager {
		public static bool SynchroniseAfterEveryKernel; // For debugging purposes

		public readonly Dictionary<string, KernelRunner> _kernelRunners = new Dictionary<string, KernelRunner>();

		public readonly CudaManager CudaManager;

		public KernelManager(CudaManager cudaManager) {
			CudaManager = cudaManager;
			PrefixArguments = new object[0];
		}

		public object[] PrefixArguments { get; set; }

		public int BlockSize { get; set; }

		public KernelRunner this[string moduleName, string kernelName = null] {
			get { return Kernel(moduleName, kernelName ?? moduleName); }
		}

		private KernelRunner Kernel(string moduleName, string kernelName) {
			KernelRunner runner;
			if (!_kernelRunners.TryGetValue(kernelName, out runner)) {
				runner = new KernelRunner(this, CudaManager.GetOrCreateCudaKernel(moduleName, kernelName));
				_kernelRunners[kernelName] = runner;
			}
			return runner;
		}
	}
}