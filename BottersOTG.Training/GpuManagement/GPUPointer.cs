using System;
using System.Runtime.InteropServices;

namespace Telogis.RouteCloud.GPUManagement {
	[StructLayout(LayoutKind.Sequential)]
	public struct GPUPointer<TValue> where TValue : struct {
		public IntPtr Value;

		public GPUPointer(IntPtr value) {
			Value = value;
		}

		public static implicit operator GPUPointer<TValue>(IntPtr value) {
			return new GPUPointer<TValue>(value);
		}

		public static implicit operator IntPtr(GPUPointer<TValue> pointer) {
			return pointer.Value;
		}
	}
}
