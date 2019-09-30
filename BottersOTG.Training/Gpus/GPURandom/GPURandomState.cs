using System;
using System.Runtime.InteropServices;

namespace Telogis.RouteCloud.Optimizers.Common.GPURandom {
	[StructLayout(LayoutKind.Sequential)]
	public struct GPURandomState {
		public uint Taus1;
		public uint Taus2;
		public uint Taus3;
		public uint Lcg;

		public static GPURandomState Generate(Random random) {
			return new GPURandomState {
				Taus1 = GenerateTaus(random),
				Taus2 = GenerateTaus(random),
				Taus3 = GenerateTaus(random),
				Lcg = GenerateLCG(random),
			};
		}

		private static uint GenerateTaus(Random random) {
			// Taus needs to be initialised with value greater than 128
			uint result;
			do {
				result = (uint)random.Next(int.MinValue, int.MaxValue);
			} while (result <= 128);
			return result;
		}

		private static uint GenerateLCG(Random random) {
			return (uint)random.Next(int.MinValue, int.MaxValue);
		}
	}
}
