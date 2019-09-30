using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BottersOTG.Intelligence.Decisions;

namespace BottersOTG.Training {
	class TrainingResult {
		public Policy Policy;
		public double WinRate;
		public bool IsImprovement;
	}
}
