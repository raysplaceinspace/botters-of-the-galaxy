using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning {
	[StructLayout(LayoutKind.Sequential)]
	unsafe struct GPUSplit {
		public float Entropy;
		public byte SplitType;
		public byte Axis;

		public int Column;

		public float SplitAttribute;
		public uint SplitCategories;
	}
}
