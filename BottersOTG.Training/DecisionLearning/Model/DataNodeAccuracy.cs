using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.Model {
	public class DataNodeAccuracy {
		public IDataNode Node;
		public float CorrectWeight;
		public float TotalWeight;
		public float Accuracy {
			get { return CorrectWeight / TotalWeight; }
		}
	}
}
