using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.Model {
	class CategoricalSplit : IDataSplit {
		public int Axis;
		public uint Categories;

		public IDataNode Left { get; set; }
		public IDataNode Right { get; set; }

		public float[] ClassDistribution { get; set; }
	}
}
