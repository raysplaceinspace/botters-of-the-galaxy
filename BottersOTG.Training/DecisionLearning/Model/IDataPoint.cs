using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.Model {
	public interface IDataPoint {
		float Weight { get; }
		byte Class { get; }
		float[] Attributes { get; }
		uint[] Categories { get; }
	}
}
