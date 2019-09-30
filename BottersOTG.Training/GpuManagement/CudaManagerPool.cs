using System;
using System.Collections.Generic;
using System.Linq;
using ManagedCuda;
using Telogis.RoutePlan.Logging;

namespace Telogis.RouteCloud.GPUManagement {
	public class CudaManagerPool {
		private readonly object _lock = new object();
		private readonly bool _preferFastest;
		private readonly List<GpuPriority> _originalGpus;
		private readonly ILogger _logger;
		private List<GpuPriority> _sortedGpus;

		public readonly int TotalNumberOfGpus;

		public int NumberOfAvailableGpus {
			get { return _originalGpus.Count; }
		}

		public CudaManagerPool(ILogger log, IList<int> enabledGpus = null, bool preferFastest = true, bool useNewContext = false) {
			_logger = log;
			_preferFastest = preferFastest;

			bool throwIfUnavailable = enabledGpus?.Count > 0;

			Exception getDevicesException = null;
			List<GpuPriority> allGpus;
			try {
				allGpus = GetPerformanceOrderedDeviceIds()
					.Select((deviceId, id) => new GpuPriority(deviceId, id) { UseNewContext = useNewContext }).ToList();

			} catch (Exception e) when (!throwIfUnavailable) {
				getDevicesException = e;
				allGpus = new List<GpuPriority>();
			}

			if (enabledGpus != null) {
				_originalGpus = enabledGpus.Select(orderId => allGpus[orderId]).ToList();
			} else {
				_originalGpus = allGpus.ToList();
			}

			if (_originalGpus.Count > 0) {
				_logger.Info("using {numUsedGpus}/{numTotalGpus} gpus with ids [{gpuOrderIds}] and deviceIds [{gpuDeviceIds}]", new {
					numUsedGpus =_originalGpus.Count,
					numTotalGpus = allGpus.Count,
					gpuOrderIds = string.Join(",", _originalGpus.Select(g => g.OrderId)),
					gpuDeviceIds = string.Join(",", _originalGpus.Select(g => g.DeviceId)),
				});

			} else {
				if (throwIfUnavailable) {
					throw new PlatformNotSupportedException("no gpus available for use");

				} else if (getDevicesException != null) {
					_logger.Info($"In non-GPU mode as the CUDA runtime was unavailable: ({getDevicesException.GetType()}) {getDevicesException.Message}", new {
						exception = getDevicesException,
					});
				} else {
					_logger.Info("In non-GPU mode as the CUDA runtime available but no GPUs were found on the system");
				}
			}

			_sortedGpus = _originalGpus.ToList();
			TotalNumberOfGpus = allGpus.Count;
		}

		public void VerifyAvailableGPUsAreWorking() {
			for (int i = 0; i < NumberOfAvailableGpus; ++i) {
				using (CudaManager cudaManager = GetCudaManagerForThread()) {
					KernelManager kernels = new KernelManager(cudaManager);
					kernels["TestKernel"].ExecuteTask();
				}
			}
		}

		public CudaManager GetCudaManagerForThread(ILogger log = null, int gpuNumber = -1) {
			int cudaManagersInUse;
			CudaManager cudaManager;
			lock (_lock) {
				_sortedGpus = GetSortedGpus(_sortedGpus, _preferFastest);
				GpuPriority gpu = gpuNumber == -1 ? _sortedGpus.First() : _originalGpus[gpuNumber];
				cudaManager = gpu.GetCudaManager(disposeAction: manager => ReleaseCudaManager(manager));
				cudaManagersInUse = gpu.CudaManagersInUse;
			}

			(log ?? _logger).Debug("acquired cuda context for gpu {gpuOrderId} (deviceId={gpuDeviceId}). Gpu now has {gpuContextCount} contexts in use", new {
				gpuOrderId = cudaManager.OrderId,
				gpuDeviceId = cudaManager.DeviceId,
				gpuContextCount = cudaManagersInUse,
			});

			cudaManager.SetCurrent();
			return cudaManager;
		}

		private void ReleaseCudaManager(CudaManager cudaManager) {
			LogUnDisposedCudaArrays(cudaManager.AllocationStats);
			int cudaManagersInUse;
			lock (_lock) {
				GpuPriority gpu = _sortedGpus.First(g => g.DeviceId == cudaManager.DeviceId);
				gpu.ReleaseCudaManager(cudaManager);
				cudaManagersInUse = gpu.CudaManagersInUse;
			}

			_logger.Debug("released cuda context for gpu {gpuOrderId} (deviceId={gpuDeviceId}). Gpu now has {gpuContextCount} contexts in use", new {
				gpuOrderId = cudaManager.OrderId,
				gpuDeviceId = cudaManager.DeviceId,
				gpuContextCount = cudaManagersInUse,
			});
		}

		private void LogUnDisposedCudaArrays(AllocationStats allocationStats) {
			if (allocationStats.ActiveAllocations.Count == 0) {
				return;
			}
			_logger.Warn("found {numUnDisposedCudaArrays} CudaArray's left un-disposed at context cleanup, totalling {unDisposedMb:F3} MB [{unDisposedCudaArrays}]", new {
				unDisposedCudaArrays = string.Join("", allocationStats.ActiveAllocations.Select(a => a.GetShortDescription() + "\n")),
				numUnDisposedCudaArrays = allocationStats.ActiveAllocations.Count,
				unDisposedMb = allocationStats.ActiveAllocations.Sum(a => a.NumBytes) / 1024.0 / 1024.0,
			});
		}

		private class GpuPriority {
			public readonly int OrderId;
			public readonly int DeviceId;
			public int CudaManagersInUse { get; private set; }

			public bool UseNewContext;

			private Queue<CudaManager> _unusedCudaManagers = new Queue<CudaManager>();

			public GpuPriority(int deviceId, int orderId) {
				DeviceId = deviceId;
				OrderId = orderId;
			}

			public CudaManager GetCudaManager(Action<CudaManager> disposeAction) {
				CudaManagersInUse += 1;
				if (_unusedCudaManagers.Count == 0 || UseNewContext) {
					return new CudaManager(OrderId, DeviceId, disposeAction);
				} else {
					return _unusedCudaManagers.Dequeue();
				}
			}

			public void ReleaseCudaManager(CudaManager cudaManager) {
				CudaManagersInUse -= 1;
				bool cudaManagerIsValid = cudaManager.IsValid();

				if (!cudaManagerIsValid || UseNewContext) {
					cudaManager.DisposeCuda();
				} else {
					cudaManager.SetInactive();
					_unusedCudaManagers.Enqueue(cudaManager);
				}

				if (!cudaManagerIsValid) {
					// an invalid CudaManager could be because
					//   a) a segfault, so we want to fail the request
					//   b) loss of power/data to GPU, so we want to fail the instance & rerun the request somewhere else

					// failing to create a CudaManager indicates case b), and will trigger a graceful shutdown of the instance
					new CudaManager(OrderId, DeviceId, x => x.DisposeCuda()).Dispose();
				}
			}
		}

		private static List<GpuPriority> GetSortedGpus(List<GpuPriority> initial, bool preferFastest) {
			IOrderedEnumerable<GpuPriority> ordering = initial.OrderBy(x => x.CudaManagersInUse);
			if (preferFastest) { // if false, then we get round-robin behaviour, since OrderBy is a stable sort
				ordering = ordering.ThenBy(x => x.OrderId);
			}
			return ordering.ToList();
		}

		private static IList<int> GetPerformanceOrderedDeviceIds() {
			var cudaDevices = new List<Tuple<int, CudaDeviceProperties>>();
			for (int i = 0; i < CudaContext.GetDeviceCount(); i++) {
				cudaDevices.Add(Tuple.Create(i, CudaContext.GetDeviceInfo(i)));
			}

			return cudaDevices
				.Where(device => device.Item2.DriverVersion.Major != 999) // remove gpu emulators
				.OrderByDescending(device => device.Item2.ComputeCapability)
				.ThenByDescending(device => DevicePerformanceValue(device.Item2))
				.Select(device => device.Item1)
				.ToList();
		}

		private static double DevicePerformanceValue(CudaDeviceProperties device) {
			return device.ClockRate * device.MultiProcessorCount * device.MaxThreadsPerMultiProcessor;
		}
	}
}
