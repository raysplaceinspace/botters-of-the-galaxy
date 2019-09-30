using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.AttributeSplitting {
	[StructLayout(LayoutKind.Sequential)]
	unsafe struct GPUAttributeDataPoint {
		public int DataPointId;
		public float Weight;
		public byte Class;
		public float Attribute;
	}
}
