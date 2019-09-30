using System;
using System.Diagnostics;
using System.IO;

namespace Telogis.RouteCloud.GPUManagement {
	public class AllocationInfo {
		public AllocationInfo(ICudaArray array, TimeSpan creationTime, StackFrame stackFrame) {
			NumElements = array.Size;
			ElementSize = array.TypeSize;
			TypeName = array.Type.Name;
			CreationTime = creationTime;
			DeletionTime = null;

			Class = stackFrame?.GetMethod()?.DeclaringType?.Name;
			Method = stackFrame?.GetMethod()?.Name;
			File = stackFrame?.GetFileName();
			Line = stackFrame?.GetFileLineNumber();
		}

		public TimeSpan CreationTime { get; }
		public TimeSpan? DeletionTime { get; set; }

		public TimeSpan? LifeTime => DeletionTime - CreationTime;

		public string Class { get; }
		public string Method { get; }
		public string File { get; }
		public int? Line { get; }

		public long NumElements { get; }
		public long ElementSize { get; }
		public long NumBytes => NumElements * ElementSize;
		public string TypeName { get; }

		public string GetShortDescription() {
			string locationString = Class == null ? "<unknown>" : $"{Class}.{Method} ({Path.GetFileName(File ?? "")}:{Line})";
			return $"size={NumElements}*{ElementSize} bytes = {NumBytes} bytes [{TypeName}] in {locationString}";
		}
	}
}
