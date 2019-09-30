using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.Model {
	class DataLeaf : IDataNode {
		public float[] ClassDistribution { get; set; }
	}
}
