using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.CategoricalSplitting {
	[StructLayout(LayoutKind.Sequential)]
	unsafe struct GPUCategoricalDataPoint {
		public int DataPointId;
		public float Weight;
		public byte Class;
		public uint Categories;
	}
}
