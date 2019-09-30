using System;

namespace Telogis.RouteCloud.Optimizers.Common.GPURandom {
	public static class GPURandom {
		public static GPURandomState[] ConstructInitialState(int size, Random random) {
			GPURandomState[] stateArray = new GPURandomState[size];
			for (int i = 0; i < stateArray.Length; ++i) {
				stateArray[i] = GPURandomState.Generate(random);
			}
			return stateArray;
		}

		public static GPURandomState ConstructInitialStateSingle(Random random) {
			return GPURandomState.Generate(random);
		}
	}
}
