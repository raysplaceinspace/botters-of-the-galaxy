using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Telogis.RouteCloud.GPUManagement;

namespace BottersOTG.Training.DecisionLearning {
	[StructLayout(LayoutKind.Sequential)]
	struct GPUDecisionLearnerContext {
		public GPUPointer<GPUDataPoint> DataPoints;
		public int NumDataPoints;
		public float TotalWeight;

		public int NumAttributeAxes;
		public int NumCategoricalAxes;

		public GPUPointer<int> DataPointIds;

		public GPUPointer<GPUNode> Nodes;
		public int CurrentLevel;

		public GPUPointer<int> OpenNodeIds;
		public int NumOpenNodes;

		public GPUPointer<int> NextOpenNodeIds;
		public int NumNextOpenNodes;

		public int MaxOpenNodes;
	}
}
