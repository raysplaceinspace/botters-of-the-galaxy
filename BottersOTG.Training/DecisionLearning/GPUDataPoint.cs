using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning {
	[StructLayout(LayoutKind.Sequential)]
	unsafe struct GPUDataPoint {
		public float Weight;
		public byte Class;
		public fixed float AllAttributes[GPUConstants.MaxAttributeAxes];
		public fixed uint AllCategories[GPUConstants.MaxCategoricalAxes];
	}
}
