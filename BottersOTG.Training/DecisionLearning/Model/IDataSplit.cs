using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BottersOTG.Training.DecisionLearning.Model {
	public interface IDataSplit : IDataNode {
		IDataNode Left { get; }
		IDataNode Right { get; }
	}
}
