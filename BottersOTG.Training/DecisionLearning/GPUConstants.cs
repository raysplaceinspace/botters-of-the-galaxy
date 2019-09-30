using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning {
	[StructLayout(LayoutKind.Sequential)]
	public struct GPUConstants {
		public const int NumThreadsPerBlock = 128;
		public const int LargeBlock = 1024;

		public const int MaxClasses = 32;
		public const int MaxLevels = 16;
		public const int MaxAttributeAxes = 40;
		public const int MaxCategoricalAxes = 8;
		public const int MaxCategories = 32; // Number of bits in uint

		public const byte SplitType_Null = 0xFF;
		public const byte SplitType_None = 0;
		public const byte SplitType_Attribute = 1;
		public const byte SplitType_Categorical = 2;

		public const int MaxSplits = 48; // >= MaxAttributeAxes + MaxCategoricalAxes
		public const int MaxNodesAtSingleLevel = 32768; // 1 << (MaxLevels - 1);
		public const int MaxTotalNodes = 65535; // (1 << MaxLevels) - 1;

		public const float RequiredImprovementToSplit = 0.001f;
	}
}
