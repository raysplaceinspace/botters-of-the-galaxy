using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.Model {
	class AttributeSplit : IDataSplit {
		public byte Axis { get; set; }
		public float SplitValue { get; set; }

		public IDataNode Left { get; set; }
		public IDataNode Right { get; set; }

		public float[] ClassDistribution { get; set; }
	}
}
