using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Telogis.RouteCloud.GPUManagement {

	/// <summary>This object is not thread-safe, so an instance should only track allocations on a single thread</summary>
	public class AllocationStats {
		public long MaxBytes { get; private set; } = 0;
		public long TotalBytes { get; private set; } = 0;
		public long CurrentBytes { get; private set; } = 0;

		public ICollection<AllocationInfo> ActiveAllocations => _activeAllocations.Values; // has IsReadOnly = true
		public IReadOnlyCollection<AllocationInfo> AllAllocations => _allAllocations;

		private Dictionary<ICudaArray, AllocationInfo> _activeAllocations = new Dictionary<ICudaArray, AllocationInfo>();
		private List<AllocationInfo> _allAllocations = new List<AllocationInfo>();
		private Stopwatch _lifetime = Stopwatch.StartNew();

		/** <summary>Internal method, only for use by <see cref="CudaArray{T}"/> constructor.</summary> */
		internal void LogArrayCreate(ICudaArray array) {
			StackFrame frame = GetCallerFrame(new StackTrace(2, fNeedFileInfo: true));

			AllocationInfo info = new AllocationInfo(array, _lifetime.Elapsed, frame);
			if (_activeAllocations.ContainsKey(array)) {
				// problem! (but probably not crash-worthy?)
			} else {
				_activeAllocations[array] = info;
				_allAllocations.Add(info);

				CurrentBytes += info.NumBytes;
				TotalBytes += info.NumBytes;
				MaxBytes = Math.Max(MaxBytes, CurrentBytes);
			}
		}

		/** <summary>Internal method, only for use by <see cref="CudaArray{T}.Dispose"/>.</summary> */
		internal void LogArrayDispose(ICudaArray array) {
			AllocationInfo info;
			if (!_activeAllocations.TryGetValue(array, out info)) {
				// problem! (but probably not crash-worthy?)
			} else {
				info.DeletionTime = _lifetime.Elapsed;
				_activeAllocations.Remove(array);

				CurrentBytes -= info.NumBytes;
			}
		}

		private StackFrame GetCallerFrame(StackTrace st) {
			for (int i = 0; i < st.FrameCount; i++) {
				StackFrame frame = st.GetFrame(i);
				// the first method up the stack that wasn't in an ICudaArray
				if (!typeof(ICudaArray).IsAssignableFrom(frame.GetMethod().DeclaringType)) {
					return frame;
				}
			}
			return null;
		}
	}
}
