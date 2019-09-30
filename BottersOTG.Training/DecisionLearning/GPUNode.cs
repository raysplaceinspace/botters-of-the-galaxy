using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning {
	[StructLayout(LayoutKind.Sequential)]
	unsafe struct GPUNode {
		public int RangeStart;
		public int RangeLength;

		public fixed float ClassDistribution[GPUConstants.MaxClasses];
		public float Entropy;
		public float TotalWeight;

		public byte SplitType;

		// Only set if splitting (not a leaf)
		public byte SplitAxis;
		public float SplitAttribute;
		public uint SplitCategories;
		public int LeftChild;
		public int RightChild;
	}
}
